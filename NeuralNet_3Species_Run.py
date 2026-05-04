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
EXPERIMENT_NAME = "FullRun_All3SpeciesCombinations"

# O2(X) /  O2(a)  /  O2(b)  / O2(Hz) / O2+(X) / O(3P)
# O(1D) / O+(gnd) / O-(gnd) /  O3(X) / O3(exc)
TARGET_SPECIES_QUEUE = [
    ["O2(a)", "O2(b)"],
]

PLACEHOLDER_SPECIES = "NONE"
FOLDER_SPECIES_SLOTS = 3

SEEDS = list(range(32, 52))

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

# Bulky result artifacts. Central saved weights are always saved when training is needed.
SAVE_PREDICTIONS_CSV = False
SAVE_LOSS_HISTORY_CSV = False
SAVE_MODEL_INFO_JSON = False
SAVE_TEST_INPUTS_CSV = False


def is_placeholder_species(species):
    return isinstance(species, str) and species.strip().upper() == PLACEHOLDER_SPECIES


def normalize_target_species(target_species):
    if not isinstance(target_species, (list, tuple)):
        raise ValueError(f"Each queued species combination must be a list/tuple. Got: {target_species}")

    if len(target_species) > FOLDER_SPECIES_SLOTS:
        raise ValueError(
            f"Each queued species combination can contain at most {FOLDER_SPECIES_SLOTS} entries. "
            f"Got {len(target_species)}: {target_species}"
        )

    cleaned = [sp.strip() if isinstance(sp, str) else sp for sp in target_species]
    kept_species = [sp for sp in cleaned if not is_placeholder_species(sp)]

    if len(kept_species) == 0:
        raise ValueError(f"Species combination must contain at least 1 real species. Got: {target_species}")

    if len(kept_species) > FOLDER_SPECIES_SLOTS:
        raise ValueError(
            f"Species combination contains too many real species. "
            f"Got {len(kept_species)}: {kept_species}"
        )

    validate_species_config(kept_species)

    display_species = kept_species + [PLACEHOLDER_SPECIES] * (FOLDER_SPECIES_SLOTS - len(kept_species))
    return display_species, kept_species


def validate_target_species_queue(target_species_queue):
    if not target_species_queue:
        raise ValueError("TARGET_SPECIES_QUEUE cannot be empty.")

    seen = set()
    for target_species in target_species_queue:
        display_species, kept_species = normalize_target_species(target_species)
        key = tuple(display_species)
        if key in seen:
            raise ValueError(f"Duplicate species combination in TARGET_SPECIES_QUEUE: {display_species}")
        seen.add(key)


def get_normalized_target_species_queue(target_species_queue):
    normalized_queue = []
    for target_species in target_species_queue:
        display_species, kept_species = normalize_target_species(target_species)
        normalized_queue.append(
            {
                "raw_species": list(target_species),
                "display_species_for_folder": display_species,
                "kept_species": kept_species,
                "folder_name": f"{FOLDER_SPECIES_SLOTS}__" + "_".join(display_species),
                "num_species_kept": len(kept_species),
            }
        )
    return normalized_queue


def run_single_configuration(
    scheme,
    experiment_name,
    species_root,
    dataset_train,
    dataset_test,
    kept_species,
    display_species_for_folder,
    hidden_size,
    seed,
):
    num_pressure_conditions = dictionary[scheme]["n_conditions"]
    output_size = len(dictionary[scheme]["k_columns"])
    removed_species = [sp for sp in ALL_SPECIES if sp not in kept_species]

    x_train, _ = dataset_train.get_data()
    input_size = int(x_train.shape[1])

    seed_root = Path(species_root) / f"seed_{seed:04d}"
    arch_dir = seed_root / arch_to_folder_name(hidden_size)
    arch_dir.mkdir(parents=True, exist_ok=True)

    model, _, loss_history, training_record = get_or_train_model(
        scheme=scheme,
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

    test_data = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
    targets, outputs, mse = evaluate_model(model, test_data, verbose=False)

    targets_scaled = targets.numpy()
    outputs_scaled = outputs.numpy()
    targets_unscaled = dataset_test.y_data_unscaled.numpy()
    outputs_unscaled = dataset_test.scaler_output[0].inverse_transform(outputs_scaled) / 1e30

    mse_unscaled = mean_squared_error(targets_unscaled, outputs_unscaled)
    rmse_unscaled = np.sqrt(mse_unscaled)

    metrics = result_metrics_dict(
        scheme=scheme,
        experiment_name=experiment_name,
        seed=seed,
        hidden_size=hidden_size,
        input_size=input_size,
        output_size=output_size,
        activation=ACTIVATION,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_pressure_conditions=num_pressure_conditions,
        num_species_total=len(ALL_SPECIES),
        num_species_kept=len(kept_species),
        kept_species=kept_species,
        removed_species=removed_species,
        num_parameters=count_parameters(model),
        training_record=training_record,
        mse=mse,
        mse_unscaled=mse_unscaled,
        rmse_unscaled=rmse_unscaled,
        targets=targets_scaled,
        outputs=outputs_scaled,
        extra={
            "display_species_for_folder": display_species_for_folder,
            "folder_species_slots": int(FOLDER_SPECIES_SLOTS),
        },
    )

    save_metrics_files(arch_dir, metrics)

    if SAVE_PREDICTIONS_CSV:
        save_predictions_csv(
            arch_dir,
            targets_scaled=targets_scaled,
            outputs_scaled=outputs_scaled,
            targets_unscaled=targets_unscaled,
            outputs_unscaled=outputs_unscaled,
        )

    if SAVE_LOSS_HISTORY_CSV:
        save_loss_history_csv(arch_dir, loss_history)

    if SAVE_MODEL_INFO_JSON:
        save_model_info_json(arch_dir, model, hidden_size, training_record=training_record)

    return metrics


def main():
    validate_target_species_queue(TARGET_SPECIES_QUEUE)

    fullrun_root = BASE_RESULTS_DIR / SCHEME / EXPERIMENT_NAME
    fullrun_root.mkdir(parents=True, exist_ok=True)

    normalized_target_species_queue = get_normalized_target_species_queue(TARGET_SPECIES_QUEUE)

    fullrun_info = {
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "saved_weights_root": str(SAVED_WEIGHTS_ROOT),
        "architectures_tested": [list(a) for a in ARCHITECTURES],
        "seeds": SEEDS,
        "activation": ACTIVATION,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "max_epochs": MAX_EPOCHS,
        "save_predictions_csv": SAVE_PREDICTIONS_CSV,
        "save_loss_history_csv": SAVE_LOSS_HISTORY_CSV,
        "save_model_info_json": SAVE_MODEL_INFO_JSON,
        "save_test_inputs_csv": SAVE_TEST_INPUTS_CSV,
        "target_species_queue_raw": TARGET_SPECIES_QUEUE,
        "target_species_queue_normalized": normalized_target_species_queue,
        "placeholder_species": PLACEHOLDER_SPECIES,
        "folder_species_slots": FOLDER_SPECIES_SLOTS,
        "seed_note": "Configured as range(32, 52), i.e. 20 seeds. Change to range(32, 53) to include seed 52.",
        "none_placeholder_note": (
            "NONE is only used for folder/display padding. "
            "It is never passed to SPECIES_MAP or used as a neural-network input species."
        ),
        "cache_policy": "load compatible saved_weights model; otherwise train and save before evaluation",
    }
    save_json(fullrun_root / "fullrun_info.json", fullrun_info)

    queue_results = []
    total_queue_runs = len(TARGET_SPECIES_QUEUE) * len(SEEDS) * len(ARCHITECTURES)
    global_completed = 0

    print(f"Saving full queue results to: {fullrun_root}")
    print(f"Queued combinations: {len(TARGET_SPECIES_QUEUE)}")
    print(f"Total planned runs: {total_queue_runs}")

    for combo_idx, target_species_raw in enumerate(TARGET_SPECIES_QUEUE, start=1):
        display_species, target_species = normalize_target_species(target_species_raw)

        experiment_name = f"{FOLDER_SPECIES_SLOTS}__" + "_".join(display_species)
        species_root = fullrun_root / experiment_name
        species_root.mkdir(parents=True, exist_ok=True)

        print("")
        print("=" * 90)
        print(
            f"Starting combination [{combo_idx}/{len(TARGET_SPECIES_QUEUE)}]: "
            f"display={display_species} | kept={target_species}"
        )
        print(f"Combination folder: {species_root}")
        print("=" * 90)

        dataset_train, dataset_test = load_datasets_with_saved_scalers(SCHEME, target_species)
        x_train, y_train = dataset_train.get_data()

        if SAVE_TEST_INPUTS_CSV:
            feature_names = build_feature_names(target_species, dictionary[SCHEME]["n_conditions"])
            x_test_unscaled, _ = dataset_test.get_unscaled_data()
            save_test_inputs_csv(species_root, x_test_unscaled.numpy(), feature_names)

        experiment_info = {
            "scheme": SCHEME,
            "experiment_name": experiment_name,
            "train_file": dictionary[SCHEME]["main_dataset"],
            "test_file": dictionary[SCHEME]["main_dataset_test"],
            "saved_weights_root": str(SAVED_WEIGHTS_ROOT),
            "num_pressure_conditions": dictionary[SCHEME]["n_conditions"],
            "num_species_total": len(ALL_SPECIES),
            "num_species_kept": len(target_species),
            "folder_species_slots": FOLDER_SPECIES_SLOTS,
            "placeholder_species": PLACEHOLDER_SPECIES,
            "target_species_raw": list(target_species_raw),
            "display_species_for_folder": display_species,
            "species_all": ALL_SPECIES,
            "kept_species": target_species,
            "removed_species": [sp for sp in ALL_SPECIES if sp not in target_species],
            "k_columns": dictionary[SCHEME]["k_columns"],
            "architectures_tested": [list(a) for a in ARCHITECTURES],
            "seeds": SEEDS,
            "activation": ACTIVATION,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "max_epochs": MAX_EPOCHS,
            "x_train_shape": list(x_train.shape),
            "y_train_shape": list(y_train.shape),
            "none_placeholder_note": (
                "NONE is only used for folder/display padding. "
                "The model was trained only with kept_species."
            ),
            "cache_policy": "load compatible saved_weights model; otherwise train and save before evaluation",
        }
        save_json(species_root / "experiment_info.json", experiment_info)

        combination_results = []
        total_combination_runs = len(ARCHITECTURES) * len(SEEDS)
        combination_completed = 0

        for seed in SEEDS:
            for hidden_size in ARCHITECTURES:
                metrics = run_single_configuration(
                    scheme=SCHEME,
                    experiment_name=experiment_name,
                    species_root=species_root,
                    dataset_train=dataset_train,
                    dataset_test=dataset_test,
                    kept_species=target_species,
                    display_species_for_folder=display_species,
                    hidden_size=hidden_size,
                    seed=seed,
                )

                combination_results.append(metrics)
                queue_results.append(metrics)

                combination_completed += 1
                global_completed += 1

                print(
                    f"[global {global_completed}/{total_queue_runs}] "
                    f"[combo {combination_completed}/{total_combination_runs}] "
                    f"kept={target_species} | "
                    f"seed={seed} | "
                    f"arch={arch_to_folder_name(hidden_size)} | "
                    f"input_size={metrics['input_size']} | "
                    f"reused={metrics['reused_saved_weights']} | "
                    f"test_mse={metrics['test_mse']:.3e}"
                )

        save_global_summary(species_root, combination_results, filename="summary.txt")
        save_summary_csv(species_root, combination_results, filename="summary.csv")
        save_seed_aggregates(species_root, combination_results)

        print(f"Finished combination: display={display_species} | kept={target_species}")

    if queue_results:
        save_summary_csv(fullrun_root, queue_results, filename="fullrun_queue_summary.csv")
        save_global_summary(fullrun_root, queue_results, filename="fullrun_queue_summary.txt")
        save_seed_aggregates(fullrun_root, queue_results, filename="fullrun_queue_seed_aggregate_summary.csv")

    print("")
    print("Full queue done.")


if __name__ == "__main__":
    main()
