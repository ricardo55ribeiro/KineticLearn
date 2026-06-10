from tqdm import tqdm

import copy
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.NeuralNetworkModels import NeuralNet
from src.config import dict as dictionary


# --------------------------------------------------------------------------------------
# Shared cache setup

BASE_RESULTS_DIR = Path("Results_NN")
SAVED_WEIGHTS_ROOT = Path("saved_weights")

SPECIES_MAP = {
    "O2(X)": [0, 11],
    "O2(a)": [1, 12],
    "O2(b)": [2, 13],
    "O2(Hz)": [3, 14],
    "O2+(X)": [4, 15],
    "O(3P)": [5, 16],
    "O(1D)": [6, 17],
    "O+(gnd)": [7, 18],
    "O-(gnd)": [8, 19],
    "O3(X)": [9, 20],
    "O3(exc)": [10, 21],
}

ALL_SPECIES = list(SPECIES_MAP.keys())


class LoadMultiPressureDatasetTorch(torch.utils.data.Dataset):
    def __init__(
        self,
        src_file,
        nspecies,
        num_pressure_conditions,
        react_idx=None,
        m_rows=None,
        columns=None,
        scaler_input=None,
        scaler_output=None,
    ):
        self.num_pressure_conditions = num_pressure_conditions

        all_data = np.loadtxt(
            src_file,
            max_rows=m_rows,
            usecols=columns,
            delimiter=None,
            comments="#",
            skiprows=0,
            dtype=np.float64,
        )
        all_data = np.atleast_2d(all_data)

        if len(all_data) % num_pressure_conditions != 0:
            raise ValueError(
                f"The number of rows in {src_file} ({len(all_data)}) is not divisible by "
                f"num_pressure_conditions ({num_pressure_conditions})."
            )

        ncolumns = all_data.shape[1]
        x_columns = np.arange(ncolumns - nspecies, ncolumns, 1)
        y_columns = react_idx
        if react_idx is None:
            y_columns = np.arange(0, ncolumns - nspecies, 1)

        raw_x_data = all_data[:, x_columns].copy()
        raw_y_data = all_data[:, y_columns].copy()

        x_data = raw_x_data.copy()
        y_data = raw_y_data * 1e30

        x_data = x_data.reshape(num_pressure_conditions, -1, x_data.shape[1])
        y_data = y_data.reshape(num_pressure_conditions, -1, y_data.shape[1])

        raw_x_data = raw_x_data.reshape(num_pressure_conditions, -1, raw_x_data.shape[1])
        raw_y_data = raw_y_data.reshape(num_pressure_conditions, -1, raw_y_data.shape[1])

        self.scaler_input = scaler_input or [
            preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)
        ]
        self.scaler_output = scaler_output or [
            preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)
        ]

        for i in range(num_pressure_conditions):
            if scaler_input is None:
                self.scaler_input[i].fit(x_data[i])
            if scaler_output is None:
                self.scaler_output[i].fit(y_data[i])

            x_data[i] = self.scaler_input[i].transform(x_data[i])
            y_data[i] = self.scaler_output[i].transform(y_data[i])

        x_data = np.transpose(x_data, (1, 0, 2)).reshape(
            -1,
            self.num_pressure_conditions * x_data.shape[-1],
        )
        y_data = y_data[0]

        raw_x_data = np.transpose(raw_x_data, (1, 0, 2)).reshape(
            -1,
            self.num_pressure_conditions * raw_x_data.shape[-1],
        )
        raw_y_data = raw_y_data[0]

        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data).float()

        self.x_data_unscaled = torch.from_numpy(raw_x_data).float()
        self.y_data_unscaled = torch.from_numpy(raw_y_data).float()

    def get_unscaled_data(self):
        return self.x_data_unscaled, self.y_data_unscaled

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

    def get_data(self):
        return self.x_data, self.y_data


# --------------------------------------------------------------------------------------
# General helpers

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_path_token(text):
    return (
        str(text)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
        .replace(" ", "")
    )


def species_config_to_name(kept_species):
    return f"{len(kept_species)}__" + "__".join(safe_path_token(sp) for sp in kept_species)


def arch_to_folder_name(hidden_size):
    return ", ".join(map(str, hidden_size))


def validate_species_config(kept_species):
    if not kept_species:
        raise ValueError("Each species configuration must contain at least one species.")

    unknown = [sp for sp in kept_species if sp not in SPECIES_MAP]
    if unknown:
        raise ValueError(
            "Unknown species name(s): "
            + ", ".join(unknown)
            + "\nValid names are: "
            + ", ".join(ALL_SPECIES)
        )

    if len(set(kept_species)) != len(kept_species):
        raise ValueError(f"Duplicate species found in configuration: {kept_species}")


def validate_all_species_configs(species_configs):
    for kept_species in species_configs:
        validate_species_config(kept_species)


def get_kept_columns(kept_species, num_pressure_conditions):
    kept_cols = []
    for p in range(num_pressure_conditions):
        for species in kept_species:
            if p >= len(SPECIES_MAP[species]):
                raise ValueError(
                    f"SPECIES_MAP for {species} does not contain pressure condition index {p}."
                )
            kept_cols.append(SPECIES_MAP[species][p])
    return kept_cols


def get_species_indices_within_condition(kept_species):
    return [SPECIES_MAP[species][0] for species in kept_species]


def build_feature_names(species_names, num_pressure_conditions):
    return [
        f"{species}_p{p + 1}"
        for p in range(num_pressure_conditions)
        for species in species_names
    ]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def moving_average(x, window=25):
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) == 0:
        return x
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(x_padded, kernel, mode="valid")


def save_json(filepath, obj):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def save_pickle(filepath, obj):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_loss_history_csv(output_dir, history):
    if history is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_loss_smooth": moving_average(history["train_loss"], window=25),
            "val_loss_smooth": moving_average(history["val_loss"], window=25),
        }
    )
    df.to_csv(output_dir / "loss_history.csv", index=False)


def load_loss_history_csv(model_dir):
    path = Path(model_dir) / "loss_history.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "train_loss" not in df.columns or "val_loss" not in df.columns:
        return None

    return {
        "train_loss": df["train_loss"].astype(float).tolist(),
        "val_loss": df["val_loss"].astype(float).tolist(),
    }


# --------------------------------------------------------------------------------------
# Shared saved-weights cache

def saved_scheme_root(scheme):
    return SAVED_WEIGHTS_ROOT / scheme


def saved_species_root(scheme, kept_species):
    return saved_scheme_root(scheme) / species_config_to_name(kept_species)


def saved_model_dir(scheme, kept_species, seed, hidden_size):
    return saved_species_root(scheme, kept_species) / f"seed_{seed:04d}" / arch_to_folder_name(hidden_size)


def saved_model_path(scheme, kept_species, seed, hidden_size):
    return saved_model_dir(scheme, kept_species, seed, hidden_size) / "model.pth"


def apply_species_subset(dataset, kept_species, num_pressure_conditions):
    kept_cols = get_kept_columns(kept_species, num_pressure_conditions)
    dataset.x_data = dataset.x_data[:, kept_cols]
    dataset.x_data_unscaled = dataset.x_data_unscaled[:, kept_cols]
    return dataset


def load_datasets_for_species(scheme, kept_species, scaler_input=None, scaler_output=None):
    validate_species_config(kept_species)

    src_file_train = dictionary[scheme]["main_dataset"]
    src_file_test = dictionary[scheme]["main_dataset_test"]
    nspecies = dictionary[scheme]["n_densities"]
    num_pressure_conditions = dictionary[scheme]["n_conditions"]

    dataset_train = LoadMultiPressureDatasetTorch(
        src_file_train,
        nspecies,
        num_pressure_conditions,
        react_idx=dictionary[scheme]["k_columns"],
        scaler_input=scaler_input,
        scaler_output=scaler_output,
    )

    dataset_test = LoadMultiPressureDatasetTorch(
        src_file_test,
        nspecies,
        num_pressure_conditions,
        react_idx=dictionary[scheme]["k_columns"],
        scaler_input=dataset_train.scaler_input,
        scaler_output=dataset_train.scaler_output,
    )

    apply_species_subset(dataset_train, kept_species, num_pressure_conditions)
    apply_species_subset(dataset_test, kept_species, num_pressure_conditions)

    return dataset_train, dataset_test


def save_species_level_metadata(scheme, kept_species, dataset_train, dataset_test):
    root = saved_species_root(scheme, kept_species)
    root.mkdir(parents=True, exist_ok=True)

    num_pressure_conditions = dictionary[scheme]["n_conditions"]
    feature_names = build_feature_names(kept_species, num_pressure_conditions)

    save_pickle(
        root / "scalers.pkl",
        {
            "scaler_input": dataset_train.scaler_input,
            "scaler_output": dataset_train.scaler_output,
        },
    )

    x_train, y_train = dataset_train.get_data()
    x_test, y_test = dataset_test.get_data()

    species_info = {
        "scheme": scheme,
        "train_file": dictionary[scheme]["main_dataset"],
        "test_file": dictionary[scheme]["main_dataset_test"],
        "k_columns": list(dictionary[scheme]["k_columns"]),
        "num_pressure_conditions": int(num_pressure_conditions),
        "num_species_total": int(len(ALL_SPECIES)),
        "num_species_kept": int(len(kept_species)),
        "species_all": ALL_SPECIES,
        "kept_species": kept_species,
        "removed_species": [sp for sp in ALL_SPECIES if sp not in kept_species],
        "feature_names": feature_names,
        "x_train_shape": list(x_train.shape),
        "y_train_shape": list(y_train.shape),
        "x_test_shape": list(x_test.shape),
        "y_test_shape": list(y_test.shape),
    }
    save_json(root / "species_info.json", species_info)


def load_species_scalers(scheme, kept_species):
    path = saved_species_root(scheme, kept_species) / "scalers.pkl"
    if path.exists():
        return load_pickle(path)
    return None


def load_datasets_with_saved_scalers(scheme, kept_species):
    scalers = load_species_scalers(scheme, kept_species)

    if scalers is None:
        dataset_train, dataset_test = load_datasets_for_species(scheme, kept_species)
        save_species_level_metadata(scheme, kept_species, dataset_train, dataset_test)
        return dataset_train, dataset_test

    dataset_train, dataset_test = load_datasets_for_species(
        scheme,
        kept_species,
        scaler_input=scalers["scaler_input"],
        scaler_output=scalers["scaler_output"],
    )
    save_species_level_metadata(scheme, kept_species, dataset_train, dataset_test)
    return dataset_train, dataset_test


def expected_model_cache_metadata(
    scheme,
    kept_species,
    hidden_size,
    seed,
    activation,
    input_size,
    output_size,
):
    return {
        "scheme": scheme,
        "kept_species": list(kept_species),
        "hidden_size": list(hidden_size),
        "seed": int(seed),
        "activation": activation,
        "input_size": int(input_size),
        "output_size": int(output_size),
        "k_columns": list(dictionary[scheme]["k_columns"]),
        "num_pressure_conditions": int(dictionary[scheme]["n_conditions"]),
    }


def cache_metadata_mismatches(info, expected):
    mismatches = []
    for key, expected_value in expected.items():
        current_value = info.get(key)
        if key in {"kept_species", "hidden_size", "k_columns"} and current_value is not None:
            current_value = list(current_value)
        if current_value != expected_value:
            mismatches.append((key, current_value, expected_value))
    return mismatches


def train_model(
    model,
    criterion,
    optimizer,
    dataloader,
    seed,
    num_epochs=100,
    patience=5,
    val_split=0.1,
    verbose_epoch_losses=False,
):
    train_len = int((1.0 - val_split) * len(dataloader.dataset))
    val_len = len(dataloader.dataset) - train_len

    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataloader.dataset,
        [train_len, val_len],
        generator=split_generator,
    )

    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=dataloader.batch_size, shuffle=False)

    best_model_wts = copy.deepcopy(model.state_dict())
    min_val_loss = np.inf
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if verbose_epoch_losses:
            print(f"Epoch {epoch + 1}, Training loss: {train_loss}, Validation loss: {val_loss}")

        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                model.load_state_dict(best_model_wts)
                return model, history

    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(model, test_data, verbose=False):
    model.eval()
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in test_data:
            outputs = model(inputs)
            all_targets.append(targets)
            all_outputs.append(outputs)

    targets = torch.cat(all_targets, dim=0)
    outputs = torch.cat(all_outputs, dim=0)

    mse = mean_squared_error(targets.numpy(), outputs.numpy())
    if verbose:
        print(f"Mean Squared Error (MSE) on the test data: {mse}")

    return targets, outputs, mse


def predict_scaled(model, x_scaled_np):
    model.eval()
    x_tensor = torch.from_numpy(np.asarray(x_scaled_np, dtype=np.float32))
    with torch.no_grad():
        outputs = model(x_tensor).cpu().numpy()
    return outputs


def compute_scaled_metrics(targets_scaled, outputs_scaled):
    targets_scaled = np.asarray(targets_scaled, dtype=np.float64)
    outputs_scaled = np.asarray(outputs_scaled, dtype=np.float64)

    squared_errors = (outputs_scaled - targets_scaled) ** 2
    metrics = {
        "test_mse_scaled": float(np.mean(squared_errors)),
        "test_rmse_scaled": float(np.sqrt(np.mean(squared_errors))),
    }

    for i in range(targets_scaled.shape[1]):
        mse_i = float(np.mean(squared_errors[:, i]))
        metrics[f"k{i + 1}_mse_scaled"] = mse_i
        metrics[f"k{i + 1}_rmse_scaled"] = float(np.sqrt(mse_i))

    return metrics


def clean_evaluate_model(model, dataset_test):
    x_test, y_test = dataset_test.get_data()
    outputs_scaled = predict_scaled(model, x_test.numpy())
    return compute_scaled_metrics(y_test.numpy(), outputs_scaled)


def get_or_train_model(
    scheme,
    kept_species,
    hidden_size,
    seed,
    activation,
    learning_rate,
    batch_size,
    max_epochs,
    patience,
    val_split,
    dataset_train,
    dataset_test,
    verbose_epoch_losses=False,
):
    x_train, y_train = dataset_train.get_data()
    input_size = int(x_train.shape[1])
    output_size = int(y_train.shape[1])

    model_dir = saved_model_dir(scheme, kept_species, seed, hidden_size)
    model_path = model_dir / "model.pth"
    info_path = model_dir / "model_info.json"

    expected = expected_model_cache_metadata(
        scheme=scheme,
        kept_species=kept_species,
        hidden_size=hidden_size,
        seed=seed,
        activation=activation,
        input_size=input_size,
        output_size=output_size,
    )

    if model_path.exists() and info_path.exists():
        info = load_json(info_path)
        mismatches = cache_metadata_mismatches(info, expected)

        if not mismatches:
            model = NeuralNet(input_size, output_size, hidden_size, activ_f=activation)
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()

            loss_history = load_loss_history_csv(model_dir)

            record = {
                "reused_saved_weights": True,
                "saved_weights_path": str(model_path),
                "training_time_s": 0.0,
                "cached_training_time_s": float(info.get("training_time_s", 0.0)),
                "epochs_ran": int(info.get("epochs_ran", 0)),
                "best_epoch": int(info.get("best_epoch", 0)),
                "final_train_loss": float(info.get("final_train_loss", np.nan)),
                "final_val_loss": float(info.get("final_val_loss", np.nan)),
                "best_val_loss": float(info.get("best_val_loss", np.nan)),
            }
            return model, info, loss_history, record

        print("Saved weights found but metadata does not match the current run. Retraining:")
        for key, current_value, expected_value in mismatches:
            print(f"  {key}: cached={current_value} | expected={expected_value}")

    model_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)

    model = NeuralNet(input_size, output_size, hidden_size, activ_f=activation)
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    start = time.time()
    model, loss_history = train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        seed=seed,
        num_epochs=max_epochs,
        patience=patience,
        val_split=val_split,
        verbose_epoch_losses=verbose_epoch_losses,
    )
    end = time.time()

    clean_metrics = clean_evaluate_model(model, dataset_test)
    training_time_s = float(end - start)

    torch.save(model.state_dict(), model_path)
    save_loss_history_csv(model_dir, loss_history)

    info = {
        **expected,
        "split_seed": int(seed),
        "shuffle_seed": int(seed),
        "weight_seed": int(seed),
        "depth": int(len(hidden_size)),
        "num_parameters": int(count_parameters(model)),
        "num_species_total": int(len(ALL_SPECIES)),
        "num_species_kept": int(len(kept_species)),
        "removed_species": [sp for sp in ALL_SPECIES if sp not in kept_species],
        "feature_names": build_feature_names(kept_species, dictionary[scheme]["n_conditions"]),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "val_split": float(val_split),
        "epochs_ran": int(len(loss_history["train_loss"])),
        "best_epoch": int(np.argmin(loss_history["val_loss"]) + 1),
        "final_train_loss": float(loss_history["train_loss"][-1]),
        "final_val_loss": float(loss_history["val_loss"][-1]),
        "best_val_loss": float(min(loss_history["val_loss"])),
        "training_time_s": training_time_s,
        **clean_metrics,
    }

    save_json(info_path, info)
    save_json(model_dir / "clean_metrics.json", clean_metrics)

    record = {
        "reused_saved_weights": False,
        "saved_weights_path": str(model_path),
        "training_time_s": training_time_s,
        "cached_training_time_s": training_time_s,
        "epochs_ran": int(len(loss_history["train_loss"])),
        "best_epoch": int(np.argmin(loss_history["val_loss"]) + 1),
        "final_train_loss": float(loss_history["train_loss"][-1]),
        "final_val_loss": float(loss_history["val_loss"][-1]),
        "best_val_loss": float(min(loss_history["val_loss"])),
    }

    return model, info, loss_history, record


def transform_selected_unscaled_to_scaled(
    x_unscaled_selected,
    kept_species,
    scaler_input,
    num_pressure_conditions,
    nspecies,
):
    x_unscaled_selected = np.asarray(x_unscaled_selected, dtype=np.float64)
    n_samples = x_unscaled_selected.shape[0]
    n_kept = len(kept_species)
    species_indices = get_species_indices_within_condition(kept_species)

    expected_features = num_pressure_conditions * n_kept
    if x_unscaled_selected.shape[1] != expected_features:
        raise ValueError(
            f"Expected x_unscaled_selected to have {expected_features} columns "
            f"({num_pressure_conditions} conditions x {n_kept} species), "
            f"but got {x_unscaled_selected.shape[1]}."
        )

    x_scaled_selected = np.zeros_like(x_unscaled_selected, dtype=np.float64)

    for p in range(num_pressure_conditions):
        start = p * n_kept
        end = (p + 1) * n_kept

        selected_block = x_unscaled_selected[:, start:end]
        full_block = np.zeros((n_samples, nspecies), dtype=np.float64)
        full_block[:, species_indices] = selected_block

        transformed_full_block = scaler_input[p].transform(full_block)
        x_scaled_selected[:, start:end] = transformed_full_block[:, species_indices]

    return x_scaled_selected


def result_metrics_dict(
    scheme,
    experiment_name,
    seed,
    hidden_size,
    input_size,
    output_size,
    activation,
    learning_rate,
    batch_size,
    num_pressure_conditions,
    num_species_total,
    num_species_kept,
    kept_species,
    removed_species,
    num_parameters,
    training_record,
    mse,
    mse_unscaled,
    rmse_unscaled,
    targets,
    outputs,
    extra=None,
):
    metrics = {
        "scheme": scheme,
        "experiment_name": experiment_name,
        "seed": int(seed),
        "split_seed": int(seed),
        "shuffle_seed": int(seed),
        "weight_seed": int(seed),
        "hidden_size": list(hidden_size),
        "depth": len(hidden_size),
        "num_parameters": int(num_parameters),
        "input_size": int(input_size),
        "output_size": int(output_size),
        "num_pressure_conditions": int(num_pressure_conditions),
        "num_species_total": int(num_species_total),
        "num_species_kept": int(num_species_kept),
        "kept_species": kept_species,
        "removed_species": removed_species,
        "activation": activation,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epochs_ran": int(training_record.get("epochs_ran", 0)),
        "best_epoch": int(training_record.get("best_epoch", 0)),
        "final_train_loss": float(training_record.get("final_train_loss", np.nan)),
        "final_val_loss": float(training_record.get("final_val_loss", np.nan)),
        "best_val_loss": float(training_record.get("best_val_loss", np.nan)),
        "training_time_s": float(training_record.get("training_time_s", 0.0)),
        "cached_training_time_s": float(training_record.get("cached_training_time_s", 0.0)),
        "reused_saved_weights": bool(training_record.get("reused_saved_weights", False)),
        "saved_weights_path": training_record.get("saved_weights_path", ""),
        "test_mse": float(mse),
        "test_rmse": float(np.sqrt(mse)),
        "test_mse_unscaled": float(mse_unscaled),
        "test_rmse_unscaled": float(rmse_unscaled),
    }

    for i in range(output_size):
        denominator = outputs[:, i].copy()
        denominator[np.abs(denominator) < 1e-9] = 1e-9
        rel_err = np.abs((outputs[:, i] - targets[:, i]) / denominator)
        metrics[f"mean_rel_error_k{i + 1}"] = float(rel_err.mean())
        metrics[f"max_rel_error_k{i + 1}"] = float(rel_err.max())

    if extra:
        metrics.update(extra)

    return metrics


def metrics_to_dataframe(all_metrics):
    df = pd.DataFrame(all_metrics)

    if "hidden_size" in df.columns:
        df["hidden_size"] = df["hidden_size"].apply(
            lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x
        )

    if "kept_species" in df.columns:
        df["kept_species"] = df["kept_species"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    if "removed_species" in df.columns:
        df["removed_species"] = df["removed_species"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    if "display_species_for_folder" in df.columns:
        df["display_species_for_folder"] = df["display_species_for_folder"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    return df


def save_summary_csv(results_root, all_metrics, filename="summary.csv"):
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    df = metrics_to_dataframe(all_metrics)
    df.to_csv(results_root / filename, index=False)


def save_seed_aggregates(results_root, all_metrics, filename="seed_aggregate_summary.csv"):
    if not all_metrics:
        return

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_metrics)
    df["hidden_size_str"] = df["hidden_size"].apply(lambda x: ", ".join(map(str, x)))

    agg_dict = {
        "test_mse": ["mean", "std", "min", "max"],
        "test_rmse": ["mean", "std", "min", "max"],
        "test_mse_unscaled": ["mean", "std", "min", "max"],
        "test_rmse_unscaled": ["mean", "std", "min", "max"],
        "training_time_s": ["mean", "std", "min", "max"],
        "cached_training_time_s": ["mean", "std", "min", "max"],
        "epochs_ran": ["mean", "std", "min", "max"],
        "best_val_loss": ["mean", "std", "min", "max"],
    }

    if "reused_saved_weights" in df.columns:
        df["reused_saved_weights_int"] = df["reused_saved_weights"].astype(int)
        agg_dict["reused_saved_weights_int"] = ["mean", "sum"]

    for i in range(int(df["output_size"].iloc[0])):
        agg_dict[f"mean_rel_error_k{i + 1}"] = ["mean", "std", "min", "max"]
        agg_dict[f"max_rel_error_k{i + 1}"] = ["mean", "std", "min", "max"]

    group_cols = ["scheme", "experiment_name", "num_species_kept", "hidden_size_str"]
    if "display_species_for_folder" in df.columns:
        df["display_species_for_folder_str"] = df["display_species_for_folder"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
        group_cols.insert(3, "display_species_for_folder_str")

    agg = df.groupby(group_cols, as_index=False).agg(agg_dict)

    agg.columns = [
        col if isinstance(col, str) else "_".join([c for c in col if c])
        for col in agg.columns.to_flat_index()
    ]

    agg.rename(
        columns={
            "hidden_size_str": "hidden_size",
            "display_species_for_folder_str": "display_species_for_folder",
            "reused_saved_weights_int_mean": "fraction_reused_saved_weights",
            "reused_saved_weights_int_sum": "num_reused_saved_weights",
        },
        inplace=True,
    )
    agg.to_csv(results_root / filename, index=False)


def save_global_summary(results_root, all_metrics, filename="summary.txt"):
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("Architecture comparison summary")
    lines.append("")

    for metrics in all_metrics:
        lines.append(f"Seed: {metrics['seed']}")
        lines.append(f"Architecture: {metrics['hidden_size']}")
        lines.append(f"  Input size: {metrics['input_size']}")
        lines.append(f"  Kept species: {metrics['kept_species']}")
        if "display_species_for_folder" in metrics:
            lines.append(f"  Display species for folder: {metrics['display_species_for_folder']}")
        lines.append(f"  Removed species: {metrics['removed_species']}")
        lines.append(f"  Reused saved weights: {metrics.get('reused_saved_weights', False)}")
        lines.append(f"  Saved weights path: {metrics.get('saved_weights_path', '')}")
        lines.append(f"  Test MSE: {metrics['test_mse']}")
        lines.append(f"  Test MSE (unscaled): {metrics['test_mse_unscaled']}")
        lines.append(f"  Current-run training time (s): {metrics['training_time_s']}")
        lines.append(f"  Cached training time (s): {metrics['cached_training_time_s']}")
        for i in range(metrics["output_size"]):
            lines.append(f"  Mean rel err k{i + 1}: {metrics[f'mean_rel_error_k{i + 1}']}")
            lines.append(f"  Max rel err k{i + 1}: {metrics[f'max_rel_error_k{i + 1}']}")
        lines.append("")

    with open(results_root / filename, "w") as f:
        f.write("\n".join(lines))


def save_metrics_files(output_dir, metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "metrics.json", metrics)

    lines = [
        f"Experiment name: {metrics['experiment_name']}",
        f"Seed: {metrics['seed']}",
        f"Scheme: {metrics['scheme']}",
        f"Hidden size: {metrics['hidden_size']}",
        f"Input size: {metrics['input_size']}",
        f"Kept species: {metrics['kept_species']}",
        f"Removed species: {metrics['removed_species']}",
        f"Reused saved weights: {metrics.get('reused_saved_weights', False)}",
        f"Saved weights path: {metrics.get('saved_weights_path', '')}",
        f"Test MSE: {metrics['test_mse']}",
        f"Test MSE (unscaled): {metrics['test_mse_unscaled']}",
        f"Current-run training time (s): {metrics['training_time_s']}",
        f"Cached training time (s): {metrics['cached_training_time_s']}",
    ]

    if "display_species_for_folder" in metrics:
        lines.insert(7, f"Display species for folder: {metrics['display_species_for_folder']}")

    for i in range(metrics["output_size"]):
        lines.append(f"Mean relative error k{i + 1}: {metrics[f'mean_rel_error_k{i + 1}']}")
        lines.append(f"Max relative error k{i + 1}: {metrics[f'max_rel_error_k{i + 1}']}")

    with open(output_dir / "metrics.txt", "w") as f:
        f.write("\n".join(lines))


def save_predictions_csv(output_dir, targets_scaled, outputs_scaled, targets_unscaled, outputs_unscaled):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {"sample_id": np.arange(len(targets_scaled))}
    n_outputs = targets_scaled.shape[1]

    for i in range(n_outputs):
        denominator = outputs_unscaled[:, i].copy()
        denominator[np.abs(denominator) < 1e-30] = 1e-30

        abs_err = np.abs(outputs_unscaled[:, i] - targets_unscaled[:, i])
        sq_err = (outputs_unscaled[:, i] - targets_unscaled[:, i]) ** 2
        rel_err = np.abs((outputs_unscaled[:, i] - targets_unscaled[:, i]) / denominator)

        data[f"k{i + 1}_true_scaled"] = targets_scaled[:, i]
        data[f"k{i + 1}_pred_scaled"] = outputs_scaled[:, i]
        data[f"k{i + 1}_true_unscaled"] = targets_unscaled[:, i]
        data[f"k{i + 1}_pred_unscaled"] = outputs_unscaled[:, i]
        data[f"k{i + 1}_abs_err"] = abs_err
        data[f"k{i + 1}_sq_err"] = sq_err
        data[f"k{i + 1}_rel_err"] = rel_err

    pd.DataFrame(data).to_csv(output_dir / "predictions.csv", index=False)


def save_model_info_json(output_dir, model, hidden_size, training_record=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = {
        "hidden_size": list(hidden_size),
        "depth": len(hidden_size),
        "num_parameters": count_parameters(model),
    }

    if training_record is not None:
        model_info.update(
            {
                "reused_saved_weights": bool(training_record.get("reused_saved_weights", False)),
                "saved_weights_path": training_record.get("saved_weights_path", ""),
                "cached_training_time_s": float(training_record.get("cached_training_time_s", 0.0)),
            }
        )

    save_json(output_dir / "model_info.json", model_info)


def save_test_inputs_csv(results_root, x_test_unscaled, feature_names):
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(x_test_unscaled, columns=feature_names)
    df.insert(0, "sample_id", np.arange(len(df)))
    df.to_csv(results_root / "test_inputs.csv", index=False)


def save_experiment_info(results_root, info):
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    save_json(results_root / "experiment_info.json", info)


# --------------------------------------------------------------------------------------
# Setup

SCHEME = "O2_novib"
EXPERIMENT_NAME = "InverseProblem_GaussianNoiseRobustness"

NOISE_STDS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
NOISE_REPEATS = 20
NOISE_BASE_SEED = 12345

SEEDS = list(range(32, 52))

# O2(X) /  O2(a)  /  O2(b)  / O2(Hz) / O2+(X) / O(3P)
# O(1D) / O+(gnd) / O-(gnd) /  O3(X) / O3(exc)
SPECIES_CONFIGS = [
    ["O2(X)", "O2(a)", "O2(b)"],
]

ARCHITECTURES = [
    (30, 30),
    (50, 50),
    (30, 30, 30),
]

ACTIVATION = "tanh"
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
MAX_EPOCHS = 5000
PATIENCE = 100
VAL_SPLIT = 0.1
VERBOSE_EPOCH_LOSSES = False

RESAMPLE_NEGATIVE_DENSITIES = True
MAX_NOISE_RESAMPLE_ATTEMPTS = 1000
CLIP_NEGATIVE_DENSITIES = False

# Optional post-processing stages. These run only after the noise CSV files are safely saved.
SAVE_NOISE_PLOTS_AFTER_RUN = True
EXPORT_REAL_K_PREDICTIONS_AFTER_RUN = True

# --------------------------------------------------------------------------------------
# Noise-plot options

NOISE_RESULTS_TIMESTAMP = None  # None -> use Noise_MSE_Results directly, or latest timestamped subfolder if present.
PLOT_MAIN_METRIC = "test_mse_scaled"
PLOT_MAIN_METRIC_LABEL = "Scaled MSE"
PLOT_USE_LOG_Y = True
PLOT_SAVE_PNG = True
PLOT_SAVE_PDF = True
PLOT_DPI = 300
PLOT_COLORS = ["green", "red", "blue"]
PLOT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
PLOT_MAX_LEGEND_COLUMNS = 2

# --------------------------------------------------------------------------------------
# Real-K export options

# If None, export all valid direct species folders found in Noise_MSE_Results.
# Otherwise, list folder names such as ["3__O2(X)__O2(a)__O2(b)"].
SPECIES_CONFIG_NAMES_TO_EXPORT = [species_config_to_name(["O2(X)", "O2(a)", "O2(b)"])]

# If None, export all architectures/noise levels/seeds/noise repeats found in noise_results.csv.
ARCHITECTURES_TO_EXPORT = None      # e.g. ["30, 30", "50, 50"]
NOISE_PERCENTS_TO_EXPORT = None     # e.g. [0.0, 1.0, 10.0]
SEEDS_TO_EXPORT = None              # e.g. [32, 33]
NOISE_REPEATS_TO_EXPORT = None      # e.g. [0, 1, 2]

SAVE_REAL_K_RAW_CSV = True
SAVE_REAL_K_AGGREGATE_CSV = True
SAVE_REAL_K_FULL_TEXT_TABLE = True
SAVE_REAL_K_EXAMPLES_REPORT = False
SAVE_REAL_K_GLOBAL_RAW_CSV = True
SAVE_REAL_K_GLOBAL_AGGREGATE_CSV = True
SAVE_REAL_K_SIMPLE_TABLES = False  # Removed/replaced by Real_Ks_Distilled outputs.
SAVE_REAL_K_DISTILLED_TABLES = True

# The full TXT table is usually most useful from the aggregate table. Set to "raw" only if needed.
REAL_K_FULL_TEXT_TABLE_SOURCE = "aggregate"  # "aggregate" or "raw"
REAL_K_REPORT_NOISE_PERCENTS = [0.0, 1.0, 10.0]
REAL_K_REPORT_INCLUDE_BEST_MEDIAN_WORST = True

# Distilled professor-facing tables: use None for all noise levels, or a list like [0.0, 1.0, 10.0].
REAL_K_DISTILLED_NOISE_PERCENTS = [0.0, 1.0, 10.0]
REAL_K_FLOAT_FORMAT = "{:.6e}"
REAL_K_PERCENT_FORMAT = "{:.3f}"
REAL_K_TXT_BLOCK_SEPARATOR_EVERY = 80


# --------------------------------------------------------------------------------------
# Noise helpers

def noise_label(noise_std):
    return f"{100.0 * noise_std:g}%"


def make_noisy_inputs_unscaled(x_clean_unscaled, noise_std, rng):
    x_clean_unscaled = np.asarray(x_clean_unscaled, dtype=np.float64)
    noise_std = float(noise_std)

    if noise_std == 0.0:
        return x_clean_unscaled.copy()

    multiplicative_noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=x_clean_unscaled.shape,
    )
    x_noisy = x_clean_unscaled * (1.0 + multiplicative_noise)

    if RESAMPLE_NEGATIVE_DENSITIES:
        negative_mask = x_noisy < 0.0
        attempts = 0

        while np.any(negative_mask):
            attempts += 1
            if attempts > MAX_NOISE_RESAMPLE_ATTEMPTS:
                raise RuntimeError(
                    f"Failed to generate non-negative noisy densities after "
                    f"{MAX_NOISE_RESAMPLE_ATTEMPTS} resampling attempts. "
                    f"noise_std={noise_std}"
                )

            new_noise = rng.normal(
                loc=0.0,
                scale=noise_std,
                size=int(np.sum(negative_mask)),
            )

            x_noisy[negative_mask] = x_clean_unscaled[negative_mask] * (1.0 + new_noise)
            negative_mask = x_noisy < 0.0

    elif CLIP_NEGATIVE_DENSITIES:
        x_noisy = np.clip(x_noisy, 0.0, None)

    return x_noisy


def run_noise_for_single_model(
    model,
    dataset_test,
    kept_species,
    hidden_size,
    seed,
    noise_std,
    noise_repeat,
    training_record,
):
    nspecies = dictionary[SCHEME]["n_densities"]
    num_pressure_conditions = dictionary[SCHEME]["n_conditions"]

    x_test_unscaled, _ = dataset_test.get_unscaled_data()
    _, y_test_scaled = dataset_test.get_data()

    rng_seed = (
        int(NOISE_BASE_SEED)
        + int(seed) * 1_000_000
        + int(noise_repeat) * 10_000
        + int(round(noise_std * 1_000_000))
    )
    rng = np.random.default_rng(rng_seed)

    x_noisy_unscaled = make_noisy_inputs_unscaled(
        x_test_unscaled.numpy(),
        noise_std=noise_std,
        rng=rng,
    )

    x_noisy_scaled = transform_selected_unscaled_to_scaled(
        x_noisy_unscaled,
        kept_species=kept_species,
        scaler_input=dataset_test.scaler_input,
        num_pressure_conditions=num_pressure_conditions,
        nspecies=nspecies,
    )

    outputs_scaled = predict_scaled(model, x_noisy_scaled)
    metrics = compute_scaled_metrics(y_test_scaled.numpy(), outputs_scaled)

    row = {
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "species_config_name": species_config_to_name(kept_species),
        "kept_species": ", ".join(kept_species),
        "num_species_kept": int(len(kept_species)),
        "input_size": int(x_noisy_scaled.shape[1]),
        "output_size": int(y_test_scaled.shape[1]),
        "hidden_size": arch_to_folder_name(hidden_size),
        "seed": int(seed),
        "noise_repeat": int(noise_repeat),
        "noise_std": float(noise_std),
        "noise_percent": float(100.0 * noise_std),
        "noise_label": noise_label(noise_std),
        "noise_rng_seed": int(rng_seed),
        "reused_saved_weights": bool(training_record.get("reused_saved_weights", False)),
        "saved_weights_path": training_record.get("saved_weights_path", ""),
        **metrics,
    }
    return row


def aggregate_noise_results(df):
    if df.empty:
        return df

    group_cols = [
        "scheme",
        "experiment_name",
        "species_config_name",
        "kept_species",
        "num_species_kept",
        "input_size",
        "output_size",
        "hidden_size",
        "noise_std",
        "noise_percent",
        "noise_label",
    ]

    metric_cols = [col for col in df.columns if col.endswith("_scaled")]
    agg_dict = {col: ["mean", "std", "min", "max"] for col in metric_cols}
    agg_dict["seed"] = ["nunique"]
    agg_dict["noise_repeat"] = ["nunique", "count"]

    if "reused_saved_weights" in df.columns:
        df = df.copy()
        df["reused_saved_weights_int"] = df["reused_saved_weights"].astype(int)
        agg_dict["reused_saved_weights_int"] = ["mean", "sum"]

    agg = df.groupby(group_cols, as_index=False).agg(agg_dict)
    agg.columns = [
        col if isinstance(col, str) else "_".join([c for c in col if c])
        for col in agg.columns.to_flat_index()
    ]

    agg.rename(
        columns={
            "seed_nunique": "num_seeds",
            "noise_repeat_nunique": "num_noise_repeats",
            "noise_repeat_count": "num_evaluations",
            "reused_saved_weights_int_mean": "fraction_reused_saved_weights",
            "reused_saved_weights_int_sum": "num_reused_saved_weights",
        },
        inplace=True,
    )
    return agg


def is_valid_direct_species_results_folder(folder):
    """Return True only for direct species-combination result folders.

    Valid examples:
        2__O2(a)__O2(b)
        3__O2(X)__O2(a)__O2(b)
        11__O2(X)__...__O3(exc)

    Invalid examples:
        Plots
        outras_combinacoes
        nao_usar_agora
        fullrun files
        nested folders inside nao_usar_agora
    """
    folder = Path(folder)

    if not folder.is_dir():
        return False

    if not (folder / "noise_results.csv").exists():
        return False

    parts = folder.name.split("__")
    if len(parts) < 2:
        return False

    try:
        n_species_from_name = int(parts[0])
    except ValueError:
        return False

    species_tokens = parts[1:]
    if n_species_from_name != len(species_tokens):
        return False

    return True


def load_direct_species_noise_results_if_valid(folder):
    """Load a species folder only if its CSV matches the folder name."""
    folder = Path(folder)

    if not is_valid_direct_species_results_folder(folder):
        return None

    noise_results_path = folder / "noise_results.csv"

    try:
        df = pd.read_csv(noise_results_path)
    except Exception as exc:
        print(f"Skipping {folder.name}: could not read noise_results.csv ({exc})")
        return None

    required_cols = [
        "scheme",
        "experiment_name",
        "species_config_name",
        "kept_species",
        "num_species_kept",
        "input_size",
        "output_size",
        "hidden_size",
        "seed",
        "noise_repeat",
        "noise_std",
        "noise_percent",
        "noise_label",
        "test_mse_scaled",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Skipping {folder.name}: noise_results.csv is missing columns: "
            + ", ".join(missing_cols)
        )
        return None

    if df.empty:
        print(f"Skipping {folder.name}: noise_results.csv is empty.")
        return None

    species_config_names = df["species_config_name"].astype(str).dropna().unique()
    if len(species_config_names) != 1:
        print(
            f"Skipping {folder.name}: expected exactly one species_config_name, "
            f"found {len(species_config_names)}."
        )
        return None

    species_config_name = species_config_names[0]
    if species_config_name != folder.name:
        print(
            f"Skipping {folder.name}: folder name does not match species_config_name "
            f"inside CSV ({species_config_name})."
        )
        return None

    return df


def rebuild_fullrun_noise_files_from_direct_species_folders(noise_results_root):
    """Rebuild fullrun CSV files from direct species-combination folders.

    Only first-level folders are used. Therefore, if a folder is moved into something
    like 'nao_usar_agora/3__...', it is automatically ignored.
    """
    noise_results_root = Path(noise_results_root)
    noise_results_root.mkdir(parents=True, exist_ok=True)

    all_species_dfs = []
    included_folders = []

    for folder in sorted(noise_results_root.iterdir(), key=lambda p: p.name):
        species_df = load_direct_species_noise_results_if_valid(folder)

        if species_df is None:
            continue

        all_species_dfs.append(species_df)
        included_folders.append(folder.name)

    if not all_species_dfs:
        print(
            "No valid direct species-combination folders found. "
            "Global fullrun files were not rebuilt."
        )
        return pd.DataFrame(), pd.DataFrame()

    full_df = pd.concat(all_species_dfs, ignore_index=True)

    # If the same species/architecture/seed/noise/repeat exists more than once,
    # keep the last one found. In normal usage this should not happen because
    # each species folder is overwritten when rerun.
    duplicate_key = [
        "scheme",
        "experiment_name",
        "species_config_name",
        "hidden_size",
        "seed",
        "noise_repeat",
        "noise_std",
    ]
    full_df = full_df.drop_duplicates(subset=duplicate_key, keep="last")

    sort_cols = [
        "num_species_kept",
        "species_config_name",
        "hidden_size",
        "seed",
        "noise_std",
        "noise_repeat",
    ]
    full_df = full_df.sort_values(sort_cols).reset_index(drop=True)

    full_df.to_csv(noise_results_root / "fullrun_noise_results.csv", index=False)

    full_agg = aggregate_noise_results(full_df)
    full_agg.to_csv(noise_results_root / "fullrun_noise_aggregate_summary.csv", index=False)

    rebuild_info = {
        "rebuild_policy": (
            "fullrun files rebuilt from valid first-level species-combination folders "
            "containing noise_results.csv"
        ),
        "included_species_folders": included_folders,
        "num_included_species_folders": int(len(included_folders)),
        "num_fullrun_rows": int(len(full_df)),
        "num_fullrun_aggregate_rows": int(len(full_agg)),
        "ignored_nested_folders": True,
    }
    save_json(noise_results_root / "fullrun_rebuild_info.json", rebuild_info)

    print("Rebuilt global fullrun noise files from direct species folders.")
    print(f"Included {len(included_folders)} species folder(s):")
    for folder_name in included_folders:
        print(f"  {folder_name}")

    return full_df, full_agg


def save_noise_results(noise_results_root, full_results):
    """Save current-run species results, then rebuild global fullrun files from folders.

    This means:
        - the current run overwrites/updates its own species folders;
        - fullrun_noise_results.csv is rebuilt from all valid direct species folders;
        - fullrun_noise_aggregate_summary.csv is rebuilt from that reconstructed fullrun table.
    """
    noise_results_root = Path(noise_results_root)
    noise_results_root.mkdir(parents=True, exist_ok=True)

    current_df = pd.DataFrame(full_results)

    if not current_df.empty:
        for species_name, species_df in current_df.groupby("species_config_name"):
            species_root = noise_results_root / str(species_name)
            species_root.mkdir(parents=True, exist_ok=True)

            species_df = species_df.sort_values(
                ["hidden_size", "seed", "noise_std", "noise_repeat"]
            ).reset_index(drop=True)

            species_df.to_csv(species_root / "noise_results.csv", index=False)

            species_agg = aggregate_noise_results(species_df)
            species_agg.to_csv(species_root / "noise_aggregate_summary.csv", index=False)
    else:
        print("Current run produced no rows. Rebuilding global fullrun files from existing folders only.")

    rebuild_fullrun_noise_files_from_direct_species_folders(noise_results_root)


def run_noise_robustness():
    validate_all_species_configs(SPECIES_CONFIGS)

    noise_results_root = BASE_RESULTS_DIR / SCHEME / EXPERIMENT_NAME / "Noise_MSE_Results"
    noise_results_root.mkdir(parents=True, exist_ok=True)

    run_info = {
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "saved_weights_root": str(SAVED_WEIGHTS_ROOT),
        "species_configs": SPECIES_CONFIGS,
        "architectures": [list(a) for a in ARCHITECTURES],
        "seeds": SEEDS,
        "noise_stds": NOISE_STDS,
        "noise_labels": [noise_label(s) for s in NOISE_STDS],
        "noise_repeats": NOISE_REPEATS,
        "noise_base_seed": NOISE_BASE_SEED,
        "main_metric": "test_mse_scaled",
        "noise_type": "multiplicative Gaussian noise applied to all selected unscaled input-density features",
        "resample_negative_densities": RESAMPLE_NEGATIVE_DENSITIES,
        "max_noise_resample_attempts": MAX_NOISE_RESAMPLE_ATTEMPTS,
        "clip_negative_densities": CLIP_NEGATIVE_DENSITIES,
        "save_noise_plots_after_run": SAVE_NOISE_PLOTS_AFTER_RUN,
        "export_real_k_predictions_after_run": EXPORT_REAL_K_PREDICTIONS_AFTER_RUN,
        "cache_policy": "load compatible saved_weights model; otherwise train and save before evaluation",
    }
    save_json(noise_results_root / "noise_run_info.json", run_info)

    full_results = []
    total_evals = (
        len(SPECIES_CONFIGS)
        * len(ARCHITECTURES)
        * len(SEEDS)
        * len(NOISE_STDS)
        * NOISE_REPEATS
    )

    with tqdm(total=total_evals, desc="Noise robustness") as pbar:
        for kept_species in SPECIES_CONFIGS:
            dataset_train, dataset_test = load_datasets_with_saved_scalers(SCHEME, kept_species)

            for seed in SEEDS:
                for hidden_size in ARCHITECTURES:
                    model, _, _, training_record = get_or_train_model(
                        scheme=SCHEME,
                        kept_species=kept_species,
                        hidden_size=hidden_size,
                        seed=seed,
                        activation=ACTIVATION,
                        learning_rate=LEARNING_RATE,
                        batch_size=BATCH_SIZE,
                        max_epochs=MAX_EPOCHS,
                        patience=PATIENCE,
                        val_split=VAL_SPLIT,
                        dataset_train=dataset_train,
                        dataset_test=dataset_test,
                        verbose_epoch_losses=VERBOSE_EPOCH_LOSSES,
                    )

                    for noise_std in NOISE_STDS:
                        for noise_repeat in range(NOISE_REPEATS):
                            row = run_noise_for_single_model(
                                model=model,
                                dataset_test=dataset_test,
                                kept_species=kept_species,
                                hidden_size=hidden_size,
                                seed=seed,
                                noise_std=noise_std,
                                noise_repeat=noise_repeat,
                                training_record=training_record,
                            )
                            full_results.append(row)

                            pbar.set_postfix(
                                species=species_config_to_name(kept_species),
                                arch=arch_to_folder_name(hidden_size),
                                seed=seed,
                                noise=row["noise_label"],
                                mse=f"{row['test_mse_scaled']:.3e}",
                            )
                            pbar.update(1)

    save_noise_results(noise_results_root, full_results)
    print(f"Saved noise robustness results to: {noise_results_root}")

    if SAVE_NOISE_PLOTS_AFTER_RUN:
        try:
            save_noise_robustness_plots(noise_results_root)
        except Exception as exc:
            print(f"WARNING: Noise plotting failed after results were saved: {exc}")

    if EXPORT_REAL_K_PREDICTIONS_AFTER_RUN:
        try:
            export_real_k_predictions(noise_results_root)
        except Exception as exc:
            print(f"WARNING: Real-K export failed after results were saved: {exc}")


# ======================================================================================
# Optional post-processing: noise robustness plots
# ======================================================================================


def plot_safe_path_token(text):
    return (
        str(text)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
        .replace(" ", "")
        .replace(",", "_")
    )


def has_noise_results(folder):
    folder = Path(folder)
    return (
        (folder / "fullrun_noise_aggregate_summary.csv").exists()
        or (folder / "fullrun_noise_results.csv").exists()
    )


def resolve_noise_results_root(noise_results_root=None):
    parent = Path(noise_results_root) if noise_results_root is not None else BASE_RESULTS_DIR / SCHEME / EXPERIMENT_NAME / "Noise_MSE_Results"

    if not parent.exists():
        raise FileNotFoundError(
            f"Noise results folder does not exist:\n{parent}\n"
            "Run the noise robustness workflow first."
        )

    if NOISE_RESULTS_TIMESTAMP is not None:
        root = parent / NOISE_RESULTS_TIMESTAMP
        if not has_noise_results(root):
            raise FileNotFoundError(
                f"Could not find fullrun_noise_aggregate_summary.csv or "
                f"fullrun_noise_results.csv in:\n{root}"
            )
        return root

    if has_noise_results(parent):
        return parent

    candidate_roots = [p for p in parent.iterdir() if p.is_dir() and has_noise_results(p)]
    if not candidate_roots:
        raise FileNotFoundError(
            f"Could not find fullrun_noise_aggregate_summary.csv or "
            f"fullrun_noise_results.csv in:\n{parent}\n"
            "Also checked timestamped subfolders."
        )

    return sorted(candidate_roots, key=lambda p: p.name)[-1]


def flatten_noise_aggregate_if_needed(df):
    if f"{PLOT_MAIN_METRIC}_mean" in df.columns:
        return df

    if PLOT_MAIN_METRIC in df.columns:
        group_cols = [
            "scheme",
            "experiment_name",
            "species_config_name",
            "kept_species",
            "num_species_kept",
            "input_size",
            "output_size",
            "hidden_size",
            "noise_std",
            "noise_percent",
            "noise_label",
        ]
        missing_group_cols = [col for col in group_cols if col not in df.columns]
        if missing_group_cols:
            raise ValueError(
                "Raw results table is missing required grouping columns: "
                + ", ".join(missing_group_cols)
            )

        metric_cols = [col for col in df.columns if col.endswith("_scaled")]
        agg_dict = {col: ["mean", "std", "min", "max"] for col in metric_cols}
        agg = df.groupby(group_cols, as_index=False).agg(agg_dict)
        agg.columns = [
            col if isinstance(col, str) else "_".join([c for c in col if c])
            for col in agg.columns.to_flat_index()
        ]
        return agg

    raise ValueError(
        f"Could not find either {PLOT_MAIN_METRIC!r} or "
        f"{PLOT_MAIN_METRIC + '_mean'!r} in the table."
    )


def load_noise_aggregate_table(results_root):
    results_root = Path(results_root)
    agg_path = results_root / "fullrun_noise_aggregate_summary.csv"
    raw_path = results_root / "fullrun_noise_results.csv"

    if agg_path.exists():
        df = pd.read_csv(agg_path)
    elif raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        raise FileNotFoundError(
            f"Could not find aggregate or raw noise results in:\n{results_root}"
        )

    df = flatten_noise_aggregate_if_needed(df)

    required_cols = [
        "species_config_name",
        "kept_species",
        "num_species_kept",
        "hidden_size",
        "noise_percent",
        f"{PLOT_MAIN_METRIC}_mean",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns: " + ", ".join(missing_cols))

    df = df.sort_values(
        ["num_species_kept", "species_config_name", "hidden_size", "noise_percent"]
    ).reset_index(drop=True)
    return df


def plot_metric_columns(metric):
    return (
        f"{metric}_mean",
        f"{metric}_std",
        f"{metric}_min",
        f"{metric}_max",
    )


def get_plot_metric_values(df, metric):
    mean_col, std_col, _, _ = plot_metric_columns(metric)
    if mean_col not in df.columns:
        raise ValueError(f"Missing required column: {mean_col}")

    y = df[mean_col].to_numpy(dtype=float)

    if std_col in df.columns:
        yerr = df[std_col].fillna(0.0).to_numpy(dtype=float)
    else:
        yerr = np.zeros_like(y)

    return y, yerr


def short_species_label(row_or_df):
    if isinstance(row_or_df, pd.DataFrame):
        species = str(row_or_df["kept_species"].iloc[0])
    else:
        species = str(row_or_df["kept_species"])

    species = (
        species.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )
    species = ", ".join(part.strip() for part in species.split(",") if part.strip())
    return species


def ordered_unique(series):
    return list(dict.fromkeys(series.tolist()))


def plot_style_for(index):
    return {
        "color": PLOT_COLORS[index % len(PLOT_COLORS)],
        "marker": PLOT_MARKERS[index % len(PLOT_MARKERS)],
    }


def save_current_noise_figure(plt, output_base):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if PLOT_SAVE_PDF:
        plt.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    if PLOT_SAVE_PNG:
        plt.savefig(output_base.with_suffix(".png"), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def format_noise_axes(ax):
    ax.set_xlabel("Gaussian noise level on input densities (%)", fontsize=13)
    ax.set_ylabel(PLOT_MAIN_METRIC_LABEL, fontsize=13)
    if PLOT_USE_LOG_Y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)


def set_two_line_title(ax, first_line, second_line):
    ax.set_title(f"{first_line}\n{second_line}", fontsize=13)


def plot_noise_by_species(df, plots_root):
    import matplotlib.pyplot as plt

    output_root = Path(plots_root) / "by_species"
    output_root.mkdir(parents=True, exist_ok=True)

    all_architectures = ordered_unique(df["hidden_size"])
    architecture_style = {arch: plot_style_for(i) for i, arch in enumerate(all_architectures)}

    for species_config_name, species_df in df.groupby("species_config_name", sort=False):
        species_label = short_species_label(species_df)
        species_dir = output_root / plot_safe_path_token(species_config_name)
        species_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9.5, 6.5))

        for hidden_size, arch_df in species_df.groupby("hidden_size", sort=False):
            arch_df = arch_df.sort_values("noise_percent")
            x = arch_df["noise_percent"].to_numpy(dtype=float)
            y, yerr = get_plot_metric_values(arch_df, PLOT_MAIN_METRIC)
            style = architecture_style[hidden_size]

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=style["marker"],
                color=style["color"],
                linewidth=1.8,
                capsize=4,
                label=f"{hidden_size}",
            )

        set_two_line_title(
            ax,
            "Inverse noise robustness by architecture",
            f"Species: {species_label}",
        )
        ax.set_xticks(sorted(species_df["noise_percent"].unique()))
        format_noise_axes(ax)
        ax.legend(title="Architecture", fontsize=10, title_fontsize=10, frameon=False)
        fig.tight_layout()

        save_current_noise_figure(plt, species_dir / "all_architectures")

        for hidden_size, arch_df in species_df.groupby("hidden_size", sort=False):
            arch_df = arch_df.sort_values("noise_percent")
            x = arch_df["noise_percent"].to_numpy(dtype=float)
            y, yerr = get_plot_metric_values(arch_df, PLOT_MAIN_METRIC)
            style = architecture_style[hidden_size]

            fig, ax = plt.subplots(figsize=(8.5, 6.0))
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=style["marker"],
                color=style["color"],
                linewidth=1.9,
                capsize=4,
                label=f"Architecture: {hidden_size}",
            )

            set_two_line_title(
                ax,
                f"Inverse noise robustness | Architecture: {hidden_size}",
                f"Species: {species_label}",
            )
            ax.set_xticks(sorted(arch_df["noise_percent"].unique()))
            format_noise_axes(ax)
            ax.legend(fontsize=10, frameon=False)
            fig.tight_layout()

            output_name = f"architecture_{plot_safe_path_token(hidden_size)}"
            save_current_noise_figure(plt, species_dir / output_name)


def plot_noise_by_architecture(df, plots_root):
    import matplotlib.pyplot as plt

    output_dir = Path(plots_root) / "by_architecture"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_species_configs = ordered_unique(df["species_config_name"])
    species_style = {name: plot_style_for(i) for i, name in enumerate(all_species_configs)}

    for hidden_size, arch_df_all in df.groupby("hidden_size", sort=False):
        print(f"\nArchitecture: {hidden_size}")
        print("Number of species combinations:", arch_df_all["species_config_name"].nunique())
        for name in arch_df_all["species_config_name"].drop_duplicates():
            print("  ", name)

        fig, ax = plt.subplots(figsize=(10.5, 7.4))

        for species_config_name, species_df in arch_df_all.groupby("species_config_name", sort=False):
            species_df = species_df.sort_values("noise_percent")
            x = species_df["noise_percent"].to_numpy(dtype=float)
            y, yerr = get_plot_metric_values(species_df, PLOT_MAIN_METRIC)
            species_label = short_species_label(species_df)
            style = species_style[species_config_name]

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=style["marker"],
                color=style["color"],
                linewidth=1.6,
                capsize=4,
                label=species_label,
            )

        set_two_line_title(
            ax,
            "Inverse noise robustness by species configuration",
            f"Architecture: {hidden_size}",
        )
        ax.set_xticks(sorted(arch_df_all["noise_percent"].unique()))
        format_noise_axes(ax)
        ax.legend(
            title="Input species",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            fontsize=8,
            title_fontsize=9,
            frameon=False,
            ncol=PLOT_MAX_LEGEND_COLUMNS,
            handlelength=2.0,
            columnspacing=1.0,
            labelspacing=0.45,
            borderaxespad=0.0,
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.32)

        output_base = output_dir / f"architecture_{plot_safe_path_token(hidden_size)}__species_configs"
        save_current_noise_figure(plt, output_base)


def save_noise_plot_manifest(results_root, plots_root, df):
    manifest = {
        "results_root": str(results_root),
        "plots_root": str(plots_root),
        "main_metric": PLOT_MAIN_METRIC,
        "main_metric_label": PLOT_MAIN_METRIC_LABEL,
        "use_log_y": PLOT_USE_LOG_Y,
        "plot_colours": PLOT_COLORS,
        "created_folders": ["by_species", "by_architecture"],
        "overview_folder_created": False,
        "num_species_configurations": int(df["species_config_name"].nunique()),
        "num_architectures": int(df["hidden_size"].nunique()),
        "noise_percent_values": sorted(float(x) for x in df["noise_percent"].unique()),
    }
    pd.Series(manifest).to_json(Path(plots_root) / "plot_manifest.json", indent=4)


def save_noise_robustness_plots(noise_results_root=None):
    results_root = resolve_noise_results_root(noise_results_root)
    df = load_noise_aggregate_table(results_root)

    plots_root = results_root / "Plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    plot_noise_by_species(df, plots_root)
    plot_noise_by_architecture(df, plots_root)
    save_noise_plot_manifest(results_root, plots_root, df)

    print(f"Read noise results from:\n{results_root}")
    print(f"Saved noise plots to:\n{plots_root}")
    print("Created only these plot folders:")
    print(f"  {plots_root / 'by_species'}")
    print(f"  {plots_root / 'by_architecture'}")


# ======================================================================================
# Optional post-processing: real-K prediction exports
# ======================================================================================


def real_k_output_root():
    return BASE_RESULTS_DIR / SCHEME / EXPERIMENT_NAME / "RealKPredictions"


def real_k_distilled_output_root(output_root=None):
    output_root = real_k_output_root() if output_root is None else Path(output_root)
    return output_root / "Real_Ks_Distilled"


def parse_hidden_size(hidden_size_text):
    if isinstance(hidden_size_text, (tuple, list)):
        return tuple(int(x) for x in hidden_size_text)

    text = str(hidden_size_text).strip()
    if not text:
        raise ValueError("Empty hidden_size value.")

    if "," in text:
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())

    return (int(text),)


def parse_species_list_from_string(text):
    text = str(text).strip()
    text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    species = [part.strip() for part in text.split(",") if part.strip()]
    return species


def update_noise_reconstruction_settings_from_run_info(noise_results_root):
    global RESAMPLE_NEGATIVE_DENSITIES, MAX_NOISE_RESAMPLE_ATTEMPTS, CLIP_NEGATIVE_DENSITIES

    info_path = Path(noise_results_root) / "noise_run_info.json"
    if not info_path.exists():
        return {}

    info = load_json(info_path)
    if "resample_negative_densities" in info:
        RESAMPLE_NEGATIVE_DENSITIES = bool(info["resample_negative_densities"])
    if "max_noise_resample_attempts" in info:
        MAX_NOISE_RESAMPLE_ATTEMPTS = int(info["max_noise_resample_attempts"])
    if "clip_negative_densities" in info:
        CLIP_NEGATIVE_DENSITIES = bool(info["clip_negative_densities"])
    return info


def load_saved_model_for_real_k(scheme, kept_species, hidden_size, seed, input_size, output_size, activation="tanh"):
    model_dir = saved_model_dir(scheme, kept_species, seed, hidden_size)
    model_path = model_dir / "model.pth"
    info_path = model_dir / "model_info.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing saved model:\n{model_path}\n"
            "Run the noise robustness workflow first so the model exists in saved_weights."
        )

    if info_path.exists():
        info = load_json(info_path)
        activation = info.get("activation", activation)
        input_size = int(info.get("input_size", input_size))
        output_size = int(info.get("output_size", output_size))
        hidden_size = tuple(info.get("hidden_size", list(hidden_size)))

    model = NeuralNet(input_size, output_size, tuple(hidden_size), activ_f=activation)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def discover_real_k_species_result_folders(noise_results_root):
    noise_results_root = Path(noise_results_root)
    if not noise_results_root.exists():
        raise FileNotFoundError(
            f"Noise results root does not exist:\n{noise_results_root}\n"
            "Run the noise robustness workflow first."
        )

    folders = []
    for folder in sorted(noise_results_root.iterdir(), key=lambda p: p.name):
        if not is_valid_direct_species_results_folder(folder):
            continue
        if SPECIES_CONFIG_NAMES_TO_EXPORT is not None and folder.name not in SPECIES_CONFIG_NAMES_TO_EXPORT:
            continue
        folders.append(folder)

    if not folders:
        raise FileNotFoundError(
            f"No valid direct species-combination folders with noise_results.csv were found in:\n"
            f"{noise_results_root}"
        )

    return folders


def load_noise_results_for_real_k_species_folder(folder):
    folder = Path(folder)
    path = folder / "noise_results.csv"
    df = pd.read_csv(path)

    required_cols = [
        "scheme",
        "experiment_name",
        "species_config_name",
        "kept_species",
        "num_species_kept",
        "input_size",
        "output_size",
        "hidden_size",
        "seed",
        "noise_repeat",
        "noise_std",
        "noise_percent",
        "noise_label",
        "noise_rng_seed",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    df = df[df["species_config_name"].astype(str) == folder.name].copy()

    if ARCHITECTURES_TO_EXPORT is not None:
        df = df[df["hidden_size"].astype(str).isin([str(a) for a in ARCHITECTURES_TO_EXPORT])]
    if NOISE_PERCENTS_TO_EXPORT is not None:
        wanted = set(float(x) for x in NOISE_PERCENTS_TO_EXPORT)
        df = df[df["noise_percent"].astype(float).isin(wanted)]
    if SEEDS_TO_EXPORT is not None:
        wanted = set(int(x) for x in SEEDS_TO_EXPORT)
        df = df[df["seed"].astype(int).isin(wanted)]
    if NOISE_REPEATS_TO_EXPORT is not None:
        wanted = set(int(x) for x in NOISE_REPEATS_TO_EXPORT)
        df = df[df["noise_repeat"].astype(int).isin(wanted)]

    duplicate_key = ["hidden_size", "seed", "noise_repeat", "noise_std"]
    df = df.drop_duplicates(subset=duplicate_key, keep="last")

    return df.reset_index(drop=True)


def inverse_transform_outputs_to_real_k(dataset_test, outputs_scaled):
    outputs_scaled = np.asarray(outputs_scaled, dtype=np.float64)
    return dataset_test.scaler_output[0].inverse_transform(outputs_scaled) / 1e30


def build_raw_real_k_prediction_dataframe(base_row, kept_species, dataset_test, model):
    """Return the direct real-K predictions for one model/noise-realization row.

    This dataframe is intentionally raw: it keeps the network seed, the noise repeat,
    and the sample ID. No seed averaging, no noise-repeat averaging, and no ensemble
    post-processing are applied here.
    """
    nspecies = dictionary[SCHEME]["n_densities"]
    num_pressure_conditions = dictionary[SCHEME]["n_conditions"]

    x_test_unscaled, _ = dataset_test.get_unscaled_data()
    targets_real = dataset_test.y_data_unscaled.numpy().astype(np.float64)

    noise_std = float(base_row["noise_std"])
    rng_seed = int(base_row["noise_rng_seed"])
    rng = np.random.default_rng(rng_seed)

    x_noisy_unscaled = make_noisy_inputs_unscaled(
        x_test_unscaled.numpy(),
        noise_std=noise_std,
        rng=rng,
    )

    x_noisy_scaled = transform_selected_unscaled_to_scaled(
        x_noisy_unscaled,
        kept_species=kept_species,
        scaler_input=dataset_test.scaler_input,
        num_pressure_conditions=num_pressure_conditions,
        nspecies=nspecies,
    )

    outputs_scaled = predict_scaled(model, x_noisy_scaled)
    outputs_real = inverse_transform_outputs_to_real_k(dataset_test, outputs_scaled)

    n_samples = targets_real.shape[0]
    output_size = targets_real.shape[1]

    data = {
        "scheme": [base_row["scheme"]] * n_samples,
        "experiment_name": [base_row["experiment_name"]] * n_samples,
        "species_config_name": [base_row["species_config_name"]] * n_samples,
        "kept_species": [base_row["kept_species"]] * n_samples,
        "num_species_kept": [int(base_row["num_species_kept"])] * n_samples,
        "input_size": [int(base_row["input_size"])] * n_samples,
        "output_size": [int(base_row["output_size"])] * n_samples,
        "hidden_size": [str(base_row["hidden_size"])] * n_samples,
        "seed": [int(base_row["seed"])] * n_samples,
        "noise_repeat": [int(base_row["noise_repeat"])] * n_samples,
        "noise_std": [noise_std] * n_samples,
        "noise_percent": [float(base_row["noise_percent"])] * n_samples,
        "noise_label": [str(base_row["noise_label"])] * n_samples,
        "noise_rng_seed": [rng_seed] * n_samples,
        "sample_id": np.arange(n_samples, dtype=int),
    }

    for i in range(output_size):
        true_values = targets_real[:, i]
        pred_values = outputs_real[:, i]
        abs_error = np.abs(pred_values - true_values)
        denominator = np.where(np.abs(true_values) < 1e-300, np.nan, true_values)
        rel_error_percent = np.abs((pred_values - true_values) / denominator) * 100.0
        pred_over_true = pred_values / denominator

        k_name = f"k{i + 1}"
        data[f"{k_name}_true_real"] = true_values
        data[f"{k_name}_pred_real"] = pred_values
        data[f"{k_name}_abs_error_real"] = abs_error
        data[f"{k_name}_rel_error_percent"] = rel_error_percent
        data[f"{k_name}_pred_over_true"] = pred_over_true

    return pd.DataFrame(data)


def get_output_size_from_real_k_raw_df(raw_df):
    k_cols = [col for col in raw_df.columns if col.endswith("_true_real") and col.startswith("k")]
    return len(k_cols)


def aggregate_real_k_predictions(raw_df):
    """Legacy-style aggregate over both NN seeds and noise repeats.

    This is retained as a diagnostic file, not as the professor-facing Real-K table.
    The professor-facing files in Real_Ks_Distilled use raw single-model predictions
    and seed-only ensemble predictions instead.
    """
    if raw_df.empty:
        return raw_df.copy()

    output_size = get_output_size_from_real_k_raw_df(raw_df)

    group_cols = [
        "scheme",
        "experiment_name",
        "species_config_name",
        "kept_species",
        "num_species_kept",
        "input_size",
        "output_size",
        "hidden_size",
        "noise_std",
        "noise_percent",
        "noise_label",
        "sample_id",
    ]

    agg_dict = {
        "seed": ["nunique"],
        "noise_repeat": ["nunique", "count"],
    }

    for i in range(output_size):
        k_name = f"k{i + 1}"
        agg_dict[f"{k_name}_true_real"] = ["first"]
        agg_dict[f"{k_name}_pred_real"] = ["mean", "std", "min", "max"]
        agg_dict[f"{k_name}_abs_error_real"] = ["mean", "std", "min", "max"]
        agg_dict[f"{k_name}_rel_error_percent"] = ["mean", "std", "min", "max"]
        agg_dict[f"{k_name}_pred_over_true"] = ["mean", "std", "min", "max"]

    agg = raw_df.groupby(group_cols, as_index=False).agg(agg_dict)
    agg.columns = [
        col if isinstance(col, str) else "_".join([c for c in col if c])
        for col in agg.columns.to_flat_index()
    ]

    agg.rename(
        columns={
            "seed_nunique": "num_seeds",
            "noise_repeat_nunique": "num_noise_repeats",
            "noise_repeat_count": "num_evaluations",
        },
        inplace=True,
    )

    rename_true = {
        f"k{i + 1}_true_real_first": f"k{i + 1}_true_real"
        for i in range(output_size)
        if f"k{i + 1}_true_real_first" in agg.columns
    }
    agg.rename(columns=rename_true, inplace=True)

    rel_cols = [f"k{i + 1}_rel_error_percent_mean" for i in range(output_size)]
    agg["mean_rel_error_percent_across_k"] = agg[rel_cols].mean(axis=1)

    return agg.sort_values(
        ["num_species_kept", "species_config_name", "hidden_size", "noise_percent", "sample_id"]
    ).reset_index(drop=True)


def raw_real_k_to_single_model_long(raw_df):
    """Convert raw wide Real-K predictions to one row per K prediction."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    output_size = get_output_size_from_real_k_raw_df(raw_df)
    base_cols = [
        "scheme",
        "experiment_name",
        "species_config_name",
        "kept_species",
        "num_species_kept",
        "input_size",
        "output_size",
        "hidden_size",
        "seed",
        "noise_repeat",
        "noise_std",
        "noise_percent",
        "noise_label",
        "noise_rng_seed",
        "sample_id",
    ]

    rows = []
    for k in range(1, output_size + 1):
        k_name = f"k{k}"
        cols = base_cols + [
            f"{k_name}_true_real",
            f"{k_name}_pred_real",
            f"{k_name}_abs_error_real",
            f"{k_name}_rel_error_percent",
            f"{k_name}_pred_over_true",
        ]
        k_df = raw_df[cols].copy()
        k_df["K"] = f"K{k}"
        k_df.rename(
            columns={
                f"{k_name}_true_real": "K_real",
                f"{k_name}_pred_real": "K_predicted",
                f"{k_name}_abs_error_real": "absolute_error_real",
                f"{k_name}_rel_error_percent": "relative_error_percent",
                f"{k_name}_pred_over_true": "predicted_over_real",
            },
            inplace=True,
        )
        rows.append(k_df)

    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df[
        base_cols[:8]
        + ["K", "seed", "noise_repeat", "noise_std", "noise_percent", "noise_label", "noise_rng_seed", "sample_id"]
        + ["K_real", "K_predicted", "absolute_error_real", "relative_error_percent", "predicted_over_real"]
    ]

    return long_df.sort_values(
        ["species_config_name", "hidden_size", "noise_percent", "seed", "noise_repeat", "sample_id", "K"]
    ).reset_index(drop=True)


def aggregate_real_k_seed_ensemble_predictions(raw_df):
    """Average predictions over NN seeds only.

    Noise repeats remain separate. This answers: for this same noisy input realization,
    what does an ensemble over trained network seeds predict?
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    output_size = get_output_size_from_real_k_raw_df(raw_df)

    group_cols = [
        "scheme",
        "experiment_name",
        "species_config_name",
        "kept_species",
        "num_species_kept",
        "input_size",
        "output_size",
        "hidden_size",
        "noise_std",
        "noise_percent",
        "noise_label",
        "noise_repeat",
        "sample_id",
    ]

    agg_dict = {"seed": ["nunique", "count"]}
    for i in range(output_size):
        k_name = f"k{i + 1}"
        agg_dict[f"{k_name}_true_real"] = ["first"]
        agg_dict[f"{k_name}_pred_real"] = ["mean", "std", "min", "max"]
        agg_dict[f"{k_name}_abs_error_real"] = ["mean", "std", "min", "max"]
        agg_dict[f"{k_name}_rel_error_percent"] = ["mean", "std", "min", "max"]
        agg_dict[f"{k_name}_pred_over_true"] = ["mean", "std", "min", "max"]

    agg = raw_df.groupby(group_cols, as_index=False).agg(agg_dict)
    agg.columns = [
        col if isinstance(col, str) else "_".join([c for c in col if c])
        for col in agg.columns.to_flat_index()
    ]

    agg.rename(
        columns={
            "seed_nunique": "num_network_seeds",
            "seed_count": "num_seed_predictions",
        },
        inplace=True,
    )

    for i in range(output_size):
        k_name = f"k{i + 1}"
        first_col = f"{k_name}_true_real_first"
        if first_col in agg.columns:
            agg.rename(columns={first_col: f"{k_name}_true_real"}, inplace=True)

        true_col = f"{k_name}_true_real"
        pred_mean_col = f"{k_name}_pred_real_mean"
        ensemble_rel_col = f"{k_name}_ensemble_mean_rel_error_percent"
        ensemble_abs_col = f"{k_name}_ensemble_mean_abs_error_real"
        ensemble_ratio_col = f"{k_name}_ensemble_mean_pred_over_true"

        denominator = np.where(np.abs(agg[true_col].to_numpy(dtype=float)) < 1e-300, np.nan, agg[true_col].to_numpy(dtype=float))
        pred_mean = agg[pred_mean_col].to_numpy(dtype=float)
        true_values = agg[true_col].to_numpy(dtype=float)

        agg[ensemble_abs_col] = np.abs(pred_mean - true_values)
        agg[ensemble_rel_col] = np.abs((pred_mean - true_values) / denominator) * 100.0
        agg[ensemble_ratio_col] = pred_mean / denominator

    return agg.sort_values(
        ["num_species_kept", "species_config_name", "hidden_size", "noise_percent", "noise_repeat", "sample_id"]
    ).reset_index(drop=True)


def seed_ensemble_to_long(ensemble_df):
    if ensemble_df is None or ensemble_df.empty:
        return pd.DataFrame()

    k_indices = sorted(
        int(col.split("_")[0][1:])
        for col in ensemble_df.columns
        if col.startswith("k") and col.endswith("_true_real")
    )

    base_cols = [
        "scheme",
        "experiment_name",
        "species_config_name",
        "kept_species",
        "num_species_kept",
        "input_size",
        "output_size",
        "hidden_size",
        "noise_std",
        "noise_percent",
        "noise_label",
        "noise_repeat",
        "sample_id",
        "num_network_seeds",
        "num_seed_predictions",
    ]

    rows = []
    for k in k_indices:
        k_name = f"k{k}"
        cols = base_cols + [
            f"{k_name}_true_real",
            f"{k_name}_pred_real_mean",
            f"{k_name}_pred_real_std",
            f"{k_name}_pred_real_min",
            f"{k_name}_pred_real_max",
            f"{k_name}_ensemble_mean_abs_error_real",
            f"{k_name}_ensemble_mean_rel_error_percent",
            f"{k_name}_ensemble_mean_pred_over_true",
            f"{k_name}_rel_error_percent_mean",
            f"{k_name}_rel_error_percent_std",
            f"{k_name}_rel_error_percent_min",
            f"{k_name}_rel_error_percent_max",
        ]
        k_df = ensemble_df[cols].copy()
        k_df["K"] = f"K{k}"
        k_df.rename(
            columns={
                f"{k_name}_true_real": "K_real",
                f"{k_name}_pred_real_mean": "K_predicted_seed_mean",
                f"{k_name}_pred_real_std": "K_predicted_seed_std",
                f"{k_name}_pred_real_min": "K_predicted_seed_min",
                f"{k_name}_pred_real_max": "K_predicted_seed_max",
                f"{k_name}_ensemble_mean_abs_error_real": "ensemble_mean_absolute_error_real",
                f"{k_name}_ensemble_mean_rel_error_percent": "ensemble_mean_relative_error_percent",
                f"{k_name}_ensemble_mean_pred_over_true": "ensemble_mean_predicted_over_real",
                f"{k_name}_rel_error_percent_mean": "single_seed_relative_error_percent_mean",
                f"{k_name}_rel_error_percent_std": "single_seed_relative_error_percent_std",
                f"{k_name}_rel_error_percent_min": "single_seed_relative_error_percent_min",
                f"{k_name}_rel_error_percent_max": "single_seed_relative_error_percent_max",
            },
            inplace=True,
        )
        rows.append(k_df)

    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df[
        base_cols[:8]
        + ["K", "noise_std", "noise_percent", "noise_label", "noise_repeat", "sample_id", "num_network_seeds", "num_seed_predictions"]
        + [
            "K_real",
            "K_predicted_seed_mean",
            "K_predicted_seed_std",
            "K_predicted_seed_min",
            "K_predicted_seed_max",
            "ensemble_mean_absolute_error_real",
            "ensemble_mean_relative_error_percent",
            "ensemble_mean_predicted_over_real",
            "single_seed_relative_error_percent_mean",
            "single_seed_relative_error_percent_std",
            "single_seed_relative_error_percent_min",
            "single_seed_relative_error_percent_max",
        ]
    ]

    return long_df.sort_values(
        ["species_config_name", "hidden_size", "noise_percent", "noise_repeat", "sample_id", "K"]
    ).reset_index(drop=True)


def nearest_empirical_error_indices(n):
    """Return closest actual-row indices for Min, 25%, 50%, 75%, and Max.

    The 25/50/75 percentiles are empirical row selections, not interpolated values.
    For example, with five sorted errors [1, 2, 3, 4, 5], this returns rows 2, 3, 4
    for 25%, 50%, 75% respectively.
    """
    if n <= 0:
        return []

    specs = [
        ("Min", 0),
        ("25%", int(np.floor(0.25 * (n - 1) + 0.5))),
        ("50%", int(np.floor(0.50 * (n - 1) + 0.5))),
        ("75%", int(np.floor(0.75 * (n - 1) + 0.5))),
        ("Max", n - 1),
    ]
    return specs


def filter_distilled_noise_levels(df):
    if df is None or df.empty:
        return df
    if REAL_K_DISTILLED_NOISE_PERCENTS is None:
        return df.copy()
    wanted = np.asarray([float(x) for x in REAL_K_DISTILLED_NOISE_PERCENTS], dtype=float)
    noise_values = df["noise_percent"].astype(float).to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    for value in wanted:
        mask |= np.isclose(noise_values, value, rtol=0.0, atol=1e-9)
    return df.loc[mask].copy()


def select_distilled_rows(long_df, error_column, mode):
    if long_df is None or long_df.empty:
        return pd.DataFrame()

    work = filter_distilled_noise_levels(long_df)
    if work.empty:
        return pd.DataFrame()

    group_cols = ["species_config_name", "kept_species", "hidden_size", "noise_percent", "noise_label", "K"]
    selected = []

    for _, group in work.groupby(group_cols, sort=False):
        group = group.dropna(subset=[error_column]).copy()
        if group.empty:
            continue
        group = group.sort_values([error_column, "sample_id"]).reset_index(drop=True)
        for label, idx in nearest_empirical_error_indices(len(group)):
            row = group.iloc[int(idx)].copy()
            row["error_statistic"] = label
            row["selection_mode"] = mode
            row["selection_error_column"] = error_column
            selected.append(row)

    if not selected:
        return pd.DataFrame()

    selected_df = pd.DataFrame(selected)

    sort_cols = [
        "species_config_name",
        "K",
        "noise_percent",
        "hidden_size",
        "error_statistic",
    ]
    statistic_order = {"Min": 0, "25%": 1, "50%": 2, "75%": 3, "Max": 4}
    selected_df["__stat_order"] = selected_df["error_statistic"].map(statistic_order).fillna(99).astype(int)
    selected_df = selected_df.sort_values(
        ["species_config_name", "K", "noise_percent", "hidden_size", "__stat_order"]
    ).drop(columns=["__stat_order"]).reset_index(drop=True)

    return selected_df


def format_noise_percent_for_table(value):
    value = float(value)
    if np.isclose(value, 0.0):
        return "0%"
    if float(value).is_integer():
        return f"{int(value)}%"
    return f"{value:g}%"


def format_architecture_for_table(value):
    return f"({str(value).strip()})"


def format_real_k_table_value(value):
    if pd.isna(value):
        return "nan"
    return REAL_K_FLOAT_FORMAT.format(float(value))


def format_real_k_percent_value(value):
    if pd.isna(value):
        return "nan"
    return REAL_K_PERCENT_FORMAT.format(float(value))


def save_text_table(df, path, title=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if title:
            f.write(title.rstrip() + "\n")
            f.write("=" * min(140, len(title)) + "\n\n")

        if df is None or df.empty:
            f.write("No rows available.\n")
            return

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            260,
            "display.max_colwidth",
            160,
        ):
            f.write(df.to_string(index=False))
            f.write("\n")


def write_single_model_distilled_tables_txt(distilled_df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("Single-model distilled Real-K tables\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            "Each row is one direct prediction from one trained network seed and one noise repeat. "
            "Rows are selected by actual empirical errors: Min, 25%, 50%, 75%, Max. "
            "Percentile rows are nearest existing rows, not interpolated values.\n\n"
        )

        if distilled_df is None or distilled_df.empty:
            f.write("No rows available.\n")
            return

        for (species_name, species_text, k_name), table_group in distilled_df.groupby(
            ["species_config_name", "kept_species", "K"], sort=False
        ):
            f.write("#" * 120 + "\n")
            f.write(f"Species: {species_text} | {k_name}\n")
            f.write("#" * 120 + "\n")

            display_rows = []
            for _, row in table_group.iterrows():
                display_rows.append(
                    {
                        "Noise": format_noise_percent_for_table(row["noise_percent"]),
                        "Architecture": format_architecture_for_table(row["hidden_size"]),
                        "Error statistic": row["error_statistic"],
                        "Seed": int(row["seed"]),
                        "Noise repeat": int(row["noise_repeat"]),
                        "Sample": int(row["sample_id"]),
                        f"Real {k_name}": format_real_k_table_value(row["K_real"]),
                        f"Predicted {k_name}": format_real_k_table_value(row["K_predicted"]),
                        "Rel. error (%)": format_real_k_percent_value(row["relative_error_percent"]),
                    }
                )

            display_df = pd.DataFrame(display_rows)
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                240,
                "display.max_colwidth",
                120,
            ):
                f.write(display_df.to_string(index=False))
                f.write("\n\n")


def write_seed_ensemble_distilled_tables_txt(distilled_df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("Seed-ensemble distilled Real-K tables\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            "Each row is an ensemble prediction averaged over neural-network seeds only. "
            "Noise repeats are kept separate and are not averaged. Rows are selected by "
            "the relative error of the ensemble mean prediction: Min, 25%, 50%, 75%, Max. "
            "Percentile rows are nearest existing rows, not interpolated values.\n\n"
        )

        if distilled_df is None or distilled_df.empty:
            f.write("No rows available.\n")
            return

        for (species_name, species_text, k_name), table_group in distilled_df.groupby(
            ["species_config_name", "kept_species", "K"], sort=False
        ):
            f.write("#" * 120 + "\n")
            f.write(f"Species: {species_text} | {k_name}\n")
            f.write("#" * 120 + "\n")

            display_rows = []
            for _, row in table_group.iterrows():
                display_rows.append(
                    {
                        "Noise": format_noise_percent_for_table(row["noise_percent"]),
                        "Architecture": format_architecture_for_table(row["hidden_size"]),
                        "Error statistic": row["error_statistic"],
                        "Noise repeat": int(row["noise_repeat"]),
                        "Sample": int(row["sample_id"]),
                        "NN seeds": int(row["num_network_seeds"]),
                        f"Real {k_name}": format_real_k_table_value(row["K_real"]),
                        f"Predicted {k_name}": format_real_k_table_value(row["K_predicted_seed_mean"]),
                        f"Std {k_name}": format_real_k_table_value(row["K_predicted_seed_std"]),
                        "Rel. error (%)": format_real_k_percent_value(row["ensemble_mean_relative_error_percent"]),
                    }
                )

            display_df = pd.DataFrame(display_rows)
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                260,
                "display.max_colwidth",
                120,
            ):
                f.write(display_df.to_string(index=False))
                f.write("\n\n")


def save_distilled_readme(distilled_root, files_info):
    distilled_root = Path(distilled_root)
    distilled_root.mkdir(parents=True, exist_ok=True)

    lines = [
        "Real_Ks_Distilled",
        "=================",
        "",
        "This folder contains the professor-facing real-K prediction tables.",
        "",
        "Definitions:",
        "- single_model: direct model outputs. No averaging over neural-network seeds and no averaging over noise repeats.",
        "- seed_ensemble: ensemble outputs averaged over neural-network seeds only. Noise repeats remain separate.",
        "- Error statistic: actual empirical row selected after sorting by relative error. Min, 25%, 50%, 75%, and Max are real rows, not interpolated percentile values.",
        "",
        "Important:",
        "- 0% noise repeats are identical by construction, because no noise is applied.",
        "- The parent RealKPredictions folder may also contain diagnostic raw/aggregate files.",
        "- The removed old SimpleTables output used aggregate mean predictions and is intentionally not generated here.",
        "",
        "Files:",
    ]

    for label, path in files_info:
        lines.append(f"- {label}: {Path(path).name}")

    (distilled_root / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_distilled_real_k_outputs(global_raw, output_root):
    """Save professor-facing Real-K files under RealKPredictions/Real_Ks_Distilled."""
    output_root = Path(output_root)
    distilled_root = real_k_distilled_output_root(output_root)
    distilled_root.mkdir(parents=True, exist_ok=True)

    saved_files = []

    if global_raw is None or global_raw.empty:
        save_distilled_readme(distilled_root, [])
        print("Skipping distilled Real-K outputs: global raw table is empty.")
        return []

    single_long = raw_real_k_to_single_model_long(global_raw)
    ensemble_wide = aggregate_real_k_seed_ensemble_predictions(global_raw)
    ensemble_long = seed_ensemble_to_long(ensemble_wide)

    single_all_path = distilled_root / "real_k_single_model_outputs.csv"
    single_long.to_csv(single_all_path, index=False)
    saved_files.append(("single_model all direct predictions", single_all_path))

    ensemble_all_path = distilled_root / "real_k_seed_ensemble_outputs.csv"
    ensemble_long.to_csv(ensemble_all_path, index=False)
    saved_files.append(("seed_ensemble all predictions averaged over NN seeds only", ensemble_all_path))

    single_distilled = select_distilled_rows(
        single_long,
        error_column="relative_error_percent",
        mode="single_model",
    )
    single_distilled_path = distilled_root / "real_k_single_model_distilled_rows.csv"
    single_distilled.to_csv(single_distilled_path, index=False)
    saved_files.append(("single_model distilled empirical error rows", single_distilled_path))

    single_txt_path = distilled_root / "real_k_single_model_distilled_tables.txt"
    write_single_model_distilled_tables_txt(single_distilled, single_txt_path)
    saved_files.append(("single_model readable distilled tables", single_txt_path))

    ensemble_distilled = select_distilled_rows(
        ensemble_long,
        error_column="ensemble_mean_relative_error_percent",
        mode="seed_ensemble",
    )
    ensemble_distilled_path = distilled_root / "real_k_seed_ensemble_distilled_rows.csv"
    ensemble_distilled.to_csv(ensemble_distilled_path, index=False)
    saved_files.append(("seed_ensemble distilled empirical error rows", ensemble_distilled_path))

    ensemble_txt_path = distilled_root / "real_k_seed_ensemble_distilled_tables.txt"
    write_seed_ensemble_distilled_tables_txt(ensemble_distilled, ensemble_txt_path)
    saved_files.append(("seed_ensemble readable distilled tables", ensemble_txt_path))

    distilled_info = {
        "description": "Professor-facing distilled real-K prediction outputs.",
        "output_root": str(distilled_root),
        "single_model_definition": "Direct predictions from individual trained neural networks. No averaging over network seeds or noise repeats.",
        "seed_ensemble_definition": "Predictions averaged over neural-network seeds only. Noise repeats remain separate and are not averaged.",
        "error_statistics": ["Min", "25%", "50%", "75%", "Max"],
        "percentile_policy": "Nearest actual empirical row after sorting by relative error; no interpolation.",
        "distilled_noise_percents": REAL_K_DISTILLED_NOISE_PERCENTS,
        "single_model_rows": int(len(single_long)),
        "single_model_distilled_rows": int(len(single_distilled)),
        "seed_ensemble_rows": int(len(ensemble_long)),
        "seed_ensemble_distilled_rows": int(len(ensemble_distilled)),
        "saved_files": [str(path) for _, path in saved_files],
    }
    save_json(distilled_root / "distilled_export_info.json", distilled_info)
    saved_files.append(("distilled export metadata", distilled_root / "distilled_export_info.json"))

    save_distilled_readme(distilled_root, saved_files)
    saved_files.append(("readme", distilled_root / "README.txt"))

    print("\nSaved distilled real-K outputs to:")
    print(f"  {distilled_root}")
    for label, path in saved_files:
        print(f"  {Path(path).name}  ({label})")

    return [str(path) for _, path in saved_files]


def format_real_k_float(value, percent=False):
    if pd.isna(value):
        return "nan"
    if percent:
        return REAL_K_PERCENT_FORMAT.format(float(value))
    return REAL_K_FLOAT_FORMAT.format(float(value))


def write_dataframe_as_text_table(df, path, title=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if title:
            f.write(title.rstrip() + "\n")
            f.write("=" * min(120, len(title)) + "\n\n")

        if df.empty:
            f.write("No rows available.\n")
            return

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            240,
            "display.max_colwidth",
            120,
            "display.float_format",
            lambda x: REAL_K_FLOAT_FORMAT.format(x),
        ):
            if REAL_K_TXT_BLOCK_SEPARATOR_EVERY is None or REAL_K_TXT_BLOCK_SEPARATOR_EVERY <= 0:
                f.write(df.to_string(index=False))
                f.write("\n")
            else:
                for start in range(0, len(df), REAL_K_TXT_BLOCK_SEPARATOR_EVERY):
                    block = df.iloc[start : start + REAL_K_TXT_BLOCK_SEPARATOR_EVERY]
                    f.write(f"Rows {start} to {start + len(block) - 1}\n")
                    f.write("-" * 120 + "\n")
                    f.write(block.to_string(index=False))
                    f.write("\n\n")


def export_real_k_species_folder(folder, output_root):
    folder = Path(folder)
    noise_df = load_noise_results_for_real_k_species_folder(folder)

    if noise_df.empty:
        print(f"Skipping {folder.name}: no rows after filters.")
        return None, None, None

    kept_species = parse_species_list_from_string(noise_df["kept_species"].iloc[0])
    validate_species_config(kept_species)

    species_config_name = species_config_to_name(kept_species)
    if species_config_name != folder.name:
        raise ValueError(
            f"Folder name {folder.name!r} does not match kept_species-derived name {species_config_name!r}."
        )

    _, dataset_test = load_datasets_with_saved_scalers(SCHEME, kept_species)
    x_test, y_test = dataset_test.get_data()
    input_size = int(x_test.shape[1])
    output_size = int(y_test.shape[1])

    species_output_root = Path(output_root) / "by_species" / species_config_name
    species_output_root.mkdir(parents=True, exist_ok=True)

    raw_parts = []
    model_cache = {}

    iterator = noise_df.iterrows()
    progress = tqdm(
        iterator,
        total=len(noise_df),
        desc=f"Export real K | {species_config_name}",
    )

    for _, row in progress:
        hidden_size_text = str(row["hidden_size"])
        hidden_size = parse_hidden_size(hidden_size_text)
        seed = int(row["seed"])

        model_key = (hidden_size_text, seed)
        if model_key not in model_cache:
            model_cache[model_key] = load_saved_model_for_real_k(
                SCHEME,
                kept_species,
                hidden_size,
                seed,
                input_size=input_size,
                output_size=output_size,
                activation=ACTIVATION,
            )

        model = model_cache[model_key]
        raw_df_part = build_raw_real_k_prediction_dataframe(
            base_row=row,
            kept_species=kept_species,
            dataset_test=dataset_test,
            model=model,
        )
        raw_parts.append(raw_df_part)

        progress.set_postfix(
            arch=hidden_size_text,
            seed=seed,
            noise=f"{float(row['noise_percent']):g}%",
        )

    raw_df = pd.concat(raw_parts, ignore_index=True)
    raw_df = raw_df.sort_values(
        ["hidden_size", "seed", "noise_percent", "noise_repeat", "sample_id"]
    ).reset_index(drop=True)

    agg_df = aggregate_real_k_predictions(raw_df)

    if SAVE_REAL_K_RAW_CSV:
        raw_df.to_csv(species_output_root / "real_k_predictions_raw.csv", index=False)

    if SAVE_REAL_K_AGGREGATE_CSV:
        agg_df.to_csv(species_output_root / "real_k_predictions_aggregate_by_sample.csv", index=False)

    if SAVE_REAL_K_FULL_TEXT_TABLE:
        table_df = agg_df if REAL_K_FULL_TEXT_TABLE_SOURCE == "aggregate" else raw_df
        write_dataframe_as_text_table(
            table_df,
            species_output_root / "real_k_predictions_full_table.txt",
            title=f"Full real-K prediction table | Species: {agg_df['kept_species'].iloc[0]}",
        )

    return raw_df, agg_df, species_output_root


def save_real_k_global_outputs(raw_dfs, agg_dfs, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    global_raw = pd.concat(raw_dfs, ignore_index=True) if raw_dfs else pd.DataFrame()
    global_agg = pd.concat(agg_dfs, ignore_index=True) if agg_dfs else pd.DataFrame()

    if not global_raw.empty:
        global_raw = global_raw.sort_values(
            [
                "num_species_kept",
                "species_config_name",
                "hidden_size",
                "seed",
                "noise_percent",
                "noise_repeat",
                "sample_id",
            ]
        ).reset_index(drop=True)

    if not global_agg.empty:
        global_agg = global_agg.sort_values(
            [
                "num_species_kept",
                "species_config_name",
                "hidden_size",
                "noise_percent",
                "sample_id",
            ]
        ).reset_index(drop=True)

    if SAVE_REAL_K_GLOBAL_RAW_CSV and SAVE_REAL_K_RAW_CSV and not global_raw.empty:
        global_raw.to_csv(output_root / "fullrun_real_k_predictions_raw.csv", index=False)

    if SAVE_REAL_K_GLOBAL_AGGREGATE_CSV and SAVE_REAL_K_AGGREGATE_CSV and not global_agg.empty:
        global_agg.to_csv(output_root / "fullrun_real_k_predictions_aggregate_by_sample.csv", index=False)

    if SAVE_REAL_K_FULL_TEXT_TABLE:
        if REAL_K_FULL_TEXT_TABLE_SOURCE == "raw" and not global_raw.empty:
            table_df = global_raw
        else:
            table_df = global_agg

        if table_df is not None and not table_df.empty:
            write_dataframe_as_text_table(
                table_df,
                output_root / "fullrun_real_k_predictions_full_table.txt",
                title="Full real-K prediction table | All exported species combinations",
            )

    return global_raw, global_agg


def export_real_k_predictions(noise_results_root=None):
    noise_results_root = Path(noise_results_root) if noise_results_root is not None else BASE_RESULTS_DIR / SCHEME / EXPERIMENT_NAME / "Noise_MSE_Results"
    output_root = real_k_output_root()

    update_noise_reconstruction_settings_from_run_info(noise_results_root)

    output_root.mkdir(parents=True, exist_ok=True)
    species_folders = discover_real_k_species_result_folders(noise_results_root)

    print("Exporting real-K predictions from species folders:")
    for folder in species_folders:
        print(f"  {folder.name}")
    print()

    raw_dfs = []
    agg_dfs = []
    species_output_dirs = []

    for folder in species_folders:
        raw_df, agg_df, species_output_root = export_real_k_species_folder(folder, output_root)
        if raw_df is not None and not raw_df.empty:
            raw_dfs.append(raw_df)
        if agg_df is not None and not agg_df.empty:
            agg_dfs.append(agg_df)
        if species_output_root is not None:
            species_output_dirs.append(str(species_output_root))

    global_raw, global_agg = save_real_k_global_outputs(raw_dfs, agg_dfs, output_root)

    distilled_files = []
    if SAVE_REAL_K_DISTILLED_TABLES:
        distilled_files = save_distilled_real_k_outputs(global_raw, output_root)

    export_info = {
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "noise_results_root": str(noise_results_root),
        "saved_weights_root": str(SAVED_WEIGHTS_ROOT),
        "output_root": str(output_root),
        "distilled_output_root": str(real_k_distilled_output_root(output_root)),
        "species_folders_exported": [folder.name for folder in species_folders],
        "species_output_dirs": species_output_dirs,
        "architectures_to_export": ARCHITECTURES_TO_EXPORT,
        "noise_percents_to_export": NOISE_PERCENTS_TO_EXPORT,
        "seeds_to_export": SEEDS_TO_EXPORT,
        "noise_repeats_to_export": NOISE_REPEATS_TO_EXPORT,
        "distilled_noise_percents": REAL_K_DISTILLED_NOISE_PERCENTS,
        "resample_negative_densities": RESAMPLE_NEGATIVE_DENSITIES,
        "max_noise_resample_attempts": MAX_NOISE_RESAMPLE_ATTEMPTS,
        "clip_negative_densities": CLIP_NEGATIVE_DENSITIES,
        "save_raw_csv": SAVE_REAL_K_RAW_CSV,
        "save_aggregate_csv": SAVE_REAL_K_AGGREGATE_CSV,
        "save_full_text_table": SAVE_REAL_K_FULL_TEXT_TABLE,
        "full_text_table_source": REAL_K_FULL_TEXT_TABLE_SOURCE,
        "save_distilled_tables": SAVE_REAL_K_DISTILLED_TABLES,
        "distilled_files": distilled_files,
        "num_distilled_files": int(len(distilled_files)),
        "simple_tables_removed": True,
        "simple_tables_reason": "The old SimpleTables output used aggregate mean predictions and could hide individual model errors.",
        "global_raw_rows": int(len(global_raw)),
        "global_aggregate_rows": int(len(global_agg)),
    }
    save_json(output_root / "export_info.json", export_info)

    print("\nSaved real-K prediction exports to:")
    print(output_root)
    print("\nMain files:")
    if SAVE_REAL_K_GLOBAL_RAW_CSV and SAVE_REAL_K_RAW_CSV:
        print(f"  {output_root / 'fullrun_real_k_predictions_raw.csv'}")
    if SAVE_REAL_K_GLOBAL_AGGREGATE_CSV and SAVE_REAL_K_AGGREGATE_CSV:
        print(f"  {output_root / 'fullrun_real_k_predictions_aggregate_by_sample.csv'}")
    if SAVE_REAL_K_FULL_TEXT_TABLE:
        print(f"  {output_root / 'fullrun_real_k_predictions_full_table.txt'}")
    if SAVE_REAL_K_DISTILLED_TABLES:
        print(f"  {real_k_distilled_output_root(output_root)}")


def main():
    run_noise_robustness()


if __name__ == "__main__":
    main()
