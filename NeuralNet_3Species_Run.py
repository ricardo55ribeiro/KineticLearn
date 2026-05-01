import json
import os
import copy
import time
import random
from datetime import datetime

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

        if all_data.ndim == 1:
            all_data = all_data.reshape(1, -1)

        ncolumns = len(all_data[0])
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

        self.scaler_input = scaler_input or [preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)]
        self.scaler_output = scaler_output or [preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)]

        for i in range(num_pressure_conditions):
            if scaler_input is None:
                self.scaler_input[i].fit(x_data[i])
            if scaler_output is None:
                self.scaler_output[i].fit(y_data[i])
            x_data[i] = self.scaler_input[i].transform(x_data[i])
            y_data[i] = self.scaler_output[i].transform(y_data[i])

        x_data = np.transpose(x_data, (1, 0, 2)).reshape(-1, self.num_pressure_conditions * x_data.shape[-1])
        y_data = y_data[0]

        raw_x_data = np.transpose(raw_x_data, (1, 0, 2)).reshape(-1, self.num_pressure_conditions * raw_x_data.shape[-1])
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


# ----------------------------------------------------------------------------------------
# Setup

SCHEME = "O2_novib"
EXPERIMENT_NAME = "FullRun_All3SpeciesCombinations"

# "O2(X)" / "O2(a)" / "O2(b)" / "O2(Hz)" / "O2+(X)" / "O(3P)"
# "O(1D)" / "O+(gnd)" / "O-(gnd)" / "O3(X)" / "O3(exc)"
TARGET_SPECIES_QUEUE = [
    ["O2(a)", "O2(b)"],
]

PLACEHOLDER_SPECIES = "NONE"
FOLDER_SPECIES_SLOTS = 3

# Timestamp
# - If FULLRUN_TIMESTAMP is not None, results are written there.
# - Else, if REUSE_LATEST_TIMESTAMP is True, the latest timestamp folder is reused.
# - Else, a new timestamp folder is created.
FULLRUN_TIMESTAMP = None
REUSE_LATEST_TIMESTAMP = True
OVERWRITE_EXISTING_COMBINATION = False

# If True, an already-existing combination folder is skipped instead of stopping the whole queue.
SKIP_EXISTING_COMBINATION = True

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
SEEDS = list(range(32, 52))

ARCHITECTURES = [
    (30, 30),
    (50, 50),
    (30, 30, 30),
]

# Bulky artifacts disabled by default.
SAVE_MODEL_WEIGHTS = False
SAVE_PREDICTIONS_CSV = False
SAVE_LOSS_HISTORY_CSV = False
SAVE_MODEL_INFO_JSON = False
SAVE_TEST_INPUTS_CSV = False


def is_placeholder_species(species):
    return isinstance(species, str) and species.strip().upper() == PLACEHOLDER_SPECIES


def normalize_target_species(target_species):
    """
    Converts a user-defined queued species combination into:

    display_species:
        Always length FOLDER_SPECIES_SLOTS.
        Used only for folder names and metadata.
        Example: ["O2(a)", "O2(b)", "NONE"]

    kept_species:
        Real species only.
        Used for dataset column selection and neural-network training.
        Example: ["O2(a)", "O2(b)"]
    """
    if not isinstance(target_species, (list, tuple)):
        raise ValueError(f"Each queued species combination must be a list/tuple. Got: {target_species}")

    if len(target_species) > FOLDER_SPECIES_SLOTS:
        raise ValueError(
            f"Each queued species combination can contain at most {FOLDER_SPECIES_SLOTS} entries. "
            f"Got {len(target_species)}: {target_species}"
        )

    cleaned = []
    for sp in target_species:
        if isinstance(sp, str):
            cleaned.append(sp.strip())
        else:
            cleaned.append(sp)

    kept_species = [sp for sp in cleaned if not is_placeholder_species(sp)]

    if len(kept_species) == 0:
        raise ValueError(f"Species combination must contain at least 1 real species. Got: {target_species}")

    if len(kept_species) > FOLDER_SPECIES_SLOTS:
        raise ValueError(
            f"Species combination contains too many real species. "
            f"Got {len(kept_species)}: {kept_species}"
        )

    if len(set(kept_species)) != len(kept_species):
        raise ValueError(f"Species combination contains duplicate real species: {kept_species}")

    invalid = [sp for sp in kept_species if sp not in SPECIES_MAP]
    if invalid:
        raise ValueError(f"Unknown species in species combination: {invalid}")

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


def get_kept_columns(kept_species, num_pressure_conditions):
    kept_cols = []
    for p in range(num_pressure_conditions):
        for species in kept_species:
            kept_cols.append(SPECIES_MAP[species][p])
    return kept_cols


def build_feature_names(species_names, num_pressure_conditions):
    return [
        f"{species}_p{p+1}"
        for p in range(num_pressure_conditions)
        for species in species_names
    ]


def arch_to_folder_name(hidden_size):
    return ", ".join(map(str, hidden_size))


def make_timestamp_string():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_fullrun_root(base_root, scheme, experiment_name, fixed_timestamp=None, reuse_latest=True):
    parent = os.path.join(base_root, scheme, experiment_name)
    os.makedirs(parent, exist_ok=True)

    if fixed_timestamp is not None:
        root = os.path.join(parent, fixed_timestamp)
        os.makedirs(root, exist_ok=True)
        return root

    timestamp_dirs = []
    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        if os.path.isdir(path):
            try:
                datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
                timestamp_dirs.append(name)
            except ValueError:
                pass

    if reuse_latest and timestamp_dirs:
        latest = sorted(timestamp_dirs)[-1]
        return os.path.join(parent, latest)

    root = os.path.join(parent, make_timestamp_string())
    os.makedirs(root, exist_ok=True)
    return root


def moving_average(x, window=25):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(x_padded, kernel, mode="valid")


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

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose_epoch_losses:
            print(f"Epoch {epoch+1}, Training loss: {train_loss}, Validation loss: {val_loss}")

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
    with torch.no_grad():
        for inputs, targets in test_data:
            outputs = model(inputs)

    mse = mean_squared_error(targets.numpy(), outputs.numpy())
    if verbose:
        print(f"Mean Squared Error (MSE) on the test data: {mse}")

    return targets, outputs, mse


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(filepath, obj):
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=4)


def save_loss_history_csv(arch_dir, history):
    df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_loss_smooth": moving_average(history["train_loss"], window=25),
            "val_loss_smooth": moving_average(history["val_loss"], window=25),
        }
    )
    df.to_csv(os.path.join(arch_dir, "loss_history.csv"), index=False)


def save_model_info(arch_dir, model, hidden_size):
    model_info = {
        "hidden_size": list(hidden_size),
        "depth": len(hidden_size),
        "num_parameters": count_parameters(model),
    }
    save_json(os.path.join(arch_dir, "model_info.json"), model_info)


def save_test_inputs_csv(results_root, x_test_unscaled, feature_names):
    df = pd.DataFrame(x_test_unscaled, columns=feature_names)
    df.insert(0, "sample_id", np.arange(len(df)))
    df.to_csv(os.path.join(results_root, "test_inputs.csv"), index=False)


def save_predictions_csv(arch_dir, targets_scaled, outputs_scaled, targets_unscaled, outputs_unscaled):
    data = {"sample_id": np.arange(len(targets_scaled))}
    n_outputs = targets_scaled.shape[1]

    for i in range(n_outputs):
        denominator = outputs_unscaled[:, i].copy()
        denominator[np.abs(denominator) < 1e-30] = 1e-30

        abs_err = np.abs(outputs_unscaled[:, i] - targets_unscaled[:, i])
        sq_err = (outputs_unscaled[:, i] - targets_unscaled[:, i]) ** 2
        rel_err = np.abs((outputs_unscaled[:, i] - targets_unscaled[:, i]) / denominator)

        data[f"k{i+1}_true_scaled"] = targets_scaled[:, i]
        data[f"k{i+1}_pred_scaled"] = outputs_scaled[:, i]
        data[f"k{i+1}_true_unscaled"] = targets_unscaled[:, i]
        data[f"k{i+1}_pred_unscaled"] = outputs_unscaled[:, i]
        data[f"k{i+1}_abs_err"] = abs_err
        data[f"k{i+1}_sq_err"] = sq_err
        data[f"k{i+1}_rel_err"] = rel_err

    pd.DataFrame(data).to_csv(os.path.join(arch_dir, "predictions.csv"), index=False)


def compute_metrics_dict(
    scheme,
    experiment_name,
    fullrun_timestamp,
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
    display_species_for_folder,
    folder_species_slots,
    num_parameters,
    loss_history,
    training_time,
    mse,
    mse_unscaled,
    rmse_unscaled,
    targets,
    outputs,
):
    metrics = {
        "scheme": scheme,
        "experiment_name": experiment_name,
        "fullrun_timestamp": fullrun_timestamp,
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
        "display_species_for_folder": display_species_for_folder,
        "folder_species_slots": int(folder_species_slots),
        "activation": activation,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epochs_ran": len(loss_history["train_loss"]),
        "best_epoch": int(np.argmin(loss_history["val_loss"]) + 1),
        "final_train_loss": float(loss_history["train_loss"][-1]),
        "final_val_loss": float(loss_history["val_loss"][-1]),
        "best_val_loss": float(min(loss_history["val_loss"])),
        "test_mse": float(mse),
        "test_rmse": float(np.sqrt(mse)),
        "test_mse_unscaled": float(mse_unscaled),
        "test_rmse_unscaled": float(rmse_unscaled),
        "training_time_s": float(training_time),
    }

    for i in range(output_size):
        denominator = outputs[:, i].copy()
        denominator[np.abs(denominator) < 1e-9] = 1e-9
        rel_err = np.abs((outputs[:, i] - targets[:, i]) / denominator)
        metrics[f"mean_rel_error_k{i+1}"] = float(rel_err.mean())
        metrics[f"max_rel_error_k{i+1}"] = float(rel_err.max())

    return metrics


def save_metrics_files(arch_dir, metrics):
    with open(os.path.join(arch_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    lines = [
        f"Experiment name: {metrics['experiment_name']}",
        f"Full run timestamp: {metrics['fullrun_timestamp']}",
        f"Seed: {metrics['seed']}",
        f"Scheme: {metrics['scheme']}",
        f"Hidden size: {metrics['hidden_size']}",
        f"Input size: {metrics['input_size']}",
        f"Kept species: {metrics['kept_species']}",
        f"Display species for folder: {metrics['display_species_for_folder']}",
        f"Removed species: {metrics['removed_species']}",
        f"Test MSE: {metrics['test_mse']}",
        f"Test MSE (unscaled): {metrics['test_mse_unscaled']}",
        f"Training time (s): {metrics['training_time_s']}",
    ]

    for i in range(metrics["output_size"]):
        lines.append(f"Mean relative error k{i+1}: {metrics[f'mean_rel_error_k{i+1}']}")
        lines.append(f"Max relative error k{i+1}: {metrics[f'max_rel_error_k{i+1}']}")

    with open(os.path.join(arch_dir, "metrics.txt"), "w") as f:
        f.write("\n".join(lines))


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
    df = metrics_to_dataframe(all_metrics)
    df.to_csv(os.path.join(results_root, filename), index=False)


def save_global_summary(results_root, all_metrics, filename="summary.txt"):
    lines = []
    lines.append("Architecture comparison summary")
    lines.append("")

    for metrics in all_metrics:
        lines.append(f"Seed: {metrics['seed']}")
        lines.append(f"Architecture: {metrics['hidden_size']}")
        lines.append(f"  Input size: {metrics['input_size']}")
        lines.append(f"  Kept species: {metrics['kept_species']}")
        lines.append(f"  Display species for folder: {metrics['display_species_for_folder']}")
        lines.append(f"  Removed species: {metrics['removed_species']}")
        lines.append(f"  Test MSE: {metrics['test_mse']}")
        lines.append(f"  Test MSE (unscaled): {metrics['test_mse_unscaled']}")
        lines.append(f"  Training time (s): {metrics['training_time_s']}")
        for i in range(metrics["output_size"]):
            lines.append(f"  Mean rel err k{i+1}: {metrics[f'mean_rel_error_k{i+1}']}")
            lines.append(f"  Max rel err k{i+1}: {metrics[f'max_rel_error_k{i+1}']}")
        lines.append("")

    with open(os.path.join(results_root, filename), "w") as f:
        f.write("\n".join(lines))


def save_seed_aggregates(results_root, all_metrics):
    if not all_metrics:
        return

    df = pd.DataFrame(all_metrics)
    df["hidden_size_str"] = df["hidden_size"].apply(lambda x: ", ".join(map(str, x)))

    agg_dict = {
        "test_mse": ["mean", "std", "min", "max"],
        "test_rmse": ["mean", "std", "min", "max"],
        "test_mse_unscaled": ["mean", "std", "min", "max"],
        "test_rmse_unscaled": ["mean", "std", "min", "max"],
        "training_time_s": ["mean", "std", "min", "max"],
        "epochs_ran": ["mean", "std", "min", "max"],
        "best_val_loss": ["mean", "std", "min", "max"],
    }

    for i in range(int(df["output_size"].iloc[0])):
        agg_dict[f"mean_rel_error_k{i+1}"] = ["mean", "std", "min", "max"]
        agg_dict[f"max_rel_error_k{i+1}"] = ["mean", "std", "min", "max"]

    agg = df.groupby(
        ["scheme", "experiment_name", "num_species_kept", "hidden_size_str"],
        as_index=False,
    ).agg(agg_dict)

    agg.columns = [
        col if isinstance(col, str) else "_".join([c for c in col if c])
        for col in agg.columns.to_flat_index()
    ]

    agg.rename(columns={"hidden_size_str": "hidden_size"}, inplace=True)
    agg.to_csv(os.path.join(results_root, "seed_aggregate_summary.csv"), index=False)


def load_datasets_for_species(scheme, kept_species):
    src_file_train = dictionary[scheme]["main_dataset"]
    src_file_test = dictionary[scheme]["main_dataset_test"]
    nspecies = dictionary[scheme]["n_densities"]
    num_pressure_conditions = dictionary[scheme]["n_conditions"]

    dataset_train = LoadMultiPressureDatasetTorch(
        src_file_train,
        nspecies,
        num_pressure_conditions,
        react_idx=dictionary[scheme]["k_columns"],
    )

    dataset_test = LoadMultiPressureDatasetTorch(
        src_file_test,
        nspecies,
        num_pressure_conditions,
        react_idx=dictionary[scheme]["k_columns"],
        scaler_input=dataset_train.scaler_input,
        scaler_output=dataset_train.scaler_output,
    )

    kept_cols = get_kept_columns(kept_species, num_pressure_conditions)

    dataset_train.x_data = dataset_train.x_data[:, kept_cols]
    dataset_test.x_data = dataset_test.x_data[:, kept_cols]
    dataset_train.x_data_unscaled = dataset_train.x_data_unscaled[:, kept_cols]
    dataset_test.x_data_unscaled = dataset_test.x_data_unscaled[:, kept_cols]

    return dataset_train, dataset_test


def run_single_configuration(
    scheme,
    fullrun_timestamp,
    experiment_name,
    species_root,
    dataset_train,
    dataset_test,
    kept_species,
    display_species_for_folder,
    hidden_size,
    seed,
    activation,
    learning_rate,
    batch_size,
    max_epochs,
    patience,
    verbose_epoch_losses=False,
):
    num_pressure_conditions = dictionary[scheme]["n_conditions"]
    output_size = len(dictionary[scheme]["k_columns"])
    all_species = list(SPECIES_MAP.keys())
    removed_species = [sp for sp in all_species if sp not in kept_species]

    x_train, _ = dataset_train.get_data()
    input_size = x_train.shape[1]

    seed_root = os.path.join(species_root, f"seed_{seed:04d}")
    arch_dir = os.path.join(seed_root, arch_to_folder_name(hidden_size))
    os.makedirs(arch_dir, exist_ok=True)

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
        val_split=0.1,
        verbose_epoch_losses=verbose_epoch_losses,
    )
    end = time.time()

    test_data = DataLoader(dataset_test, batch_size=len(dataset_test))
    targets, outputs, mse = evaluate_model(model, test_data, verbose=False)

    targets_scaled = targets.numpy()
    outputs_scaled = outputs.numpy()
    targets_unscaled = dataset_test.y_data_unscaled.numpy()
    outputs_unscaled = dataset_test.scaler_output[0].inverse_transform(outputs_scaled) / 1e30

    mse_unscaled = mean_squared_error(targets_unscaled, outputs_unscaled)
    rmse_unscaled = np.sqrt(mse_unscaled)

    if SAVE_MODEL_WEIGHTS:
        torch.save(model.state_dict(), os.path.join(arch_dir, "model.pth"))

    metrics = compute_metrics_dict(
        scheme=scheme,
        experiment_name=experiment_name,
        fullrun_timestamp=fullrun_timestamp,
        seed=seed,
        hidden_size=hidden_size,
        input_size=input_size,
        output_size=output_size,
        activation=activation,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_pressure_conditions=num_pressure_conditions,
        num_species_total=len(all_species),
        num_species_kept=len(kept_species),
        kept_species=kept_species,
        removed_species=removed_species,
        display_species_for_folder=display_species_for_folder,
        folder_species_slots=FOLDER_SPECIES_SLOTS,
        num_parameters=count_parameters(model),
        loss_history=loss_history,
        training_time=end - start,
        mse=mse,
        mse_unscaled=mse_unscaled,
        rmse_unscaled=rmse_unscaled,
        targets=targets_scaled,
        outputs=outputs_scaled,
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
        save_model_info(arch_dir, model, hidden_size)

    return metrics


if __name__ == "__main__":
    validate_target_species_queue(TARGET_SPECIES_QUEUE)

    activation = "tanh"
    learning_rate = 0.0001
    batch_size = 16
    max_epochs = 5000
    patience = 100
    verbose_epoch_losses = False

    fullrun_root = get_fullrun_root(
        base_root="Results_NN",
        scheme=SCHEME,
        experiment_name=EXPERIMENT_NAME,
        fixed_timestamp=FULLRUN_TIMESTAMP,
        reuse_latest=REUSE_LATEST_TIMESTAMP,
    )

    fullrun_timestamp = os.path.basename(fullrun_root)

    normalized_target_species_queue = get_normalized_target_species_queue(TARGET_SPECIES_QUEUE)

    fullrun_info = {
        "scheme": SCHEME,
        "fullrun_timestamp": fullrun_timestamp,
        "experiment_name": EXPERIMENT_NAME,
        "architectures_tested": [list(a) for a in ARCHITECTURES],
        "seeds": SEEDS,
        "activation": activation,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "patience": patience,
        "max_epochs": max_epochs,
        "save_model_weights": SAVE_MODEL_WEIGHTS,
        "save_predictions_csv": SAVE_PREDICTIONS_CSV,
        "save_loss_history_csv": SAVE_LOSS_HISTORY_CSV,
        "save_model_info_json": SAVE_MODEL_INFO_JSON,
        "save_test_inputs_csv": SAVE_TEST_INPUTS_CSV,
        "reuse_latest_timestamp": REUSE_LATEST_TIMESTAMP,
        "fixed_timestamp": FULLRUN_TIMESTAMP,
        "target_species_queue_raw": TARGET_SPECIES_QUEUE,
        "target_species_queue_normalized": normalized_target_species_queue,
        "placeholder_species": PLACEHOLDER_SPECIES,
        "folder_species_slots": FOLDER_SPECIES_SLOTS,
        "skip_existing_combination": SKIP_EXISTING_COMBINATION,
        "overwrite_existing_combination": OVERWRITE_EXISTING_COMBINATION,
        "seed_note": "Configured as range(32, 52), i.e. 20 seeds. Change to range(32, 53) to include seed 52.",
        "none_placeholder_note": (
            "NONE is only used for folder/display padding. "
            "It is never passed to SPECIES_MAP or used as a neural-network input species."
        ),
    }
    save_json(os.path.join(fullrun_root, "fullrun_info.json"), fullrun_info)

    queue_results = []

    total_queue_runs = len(TARGET_SPECIES_QUEUE) * len(SEEDS) * len(ARCHITECTURES)
    global_completed = 0

    print(f"Saving full queue results to: {fullrun_root}")
    print(f"Timestamp/session: {fullrun_timestamp}")
    print(f"Queued combinations: {len(TARGET_SPECIES_QUEUE)}")
    print(f"Total planned runs: {total_queue_runs}")

    for combo_idx, target_species_raw in enumerate(TARGET_SPECIES_QUEUE, start=1):
        display_species, target_species = normalize_target_species(target_species_raw)

        experiment_name = f"{FOLDER_SPECIES_SLOTS}__" + "_".join(display_species)
        species_root = os.path.join(fullrun_root, experiment_name)

        print("")
        print("=" * 90)
        print(
            f"Starting combination [{combo_idx}/{len(TARGET_SPECIES_QUEUE)}]: "
            f"display={display_species} | kept={target_species}"
        )
        print(f"Combination folder: {species_root}")
        print("=" * 90)

        if os.path.exists(species_root) and not OVERWRITE_EXISTING_COMBINATION:
            message = (
                f"Combination folder already exists: {species_root}\n"
                f"Set OVERWRITE_EXISTING_COMBINATION = True to rerun it."
            )

            if SKIP_EXISTING_COMBINATION:
                print("Skipping existing combination.")
                print(message)
                continue

            raise FileExistsError(message)

        os.makedirs(species_root, exist_ok=True)

        dataset_train, dataset_test = load_datasets_for_species(SCHEME, target_species)
        x_train, y_train = dataset_train.get_data()

        if SAVE_TEST_INPUTS_CSV:
            feature_names = build_feature_names(target_species, dictionary[SCHEME]["n_conditions"])
            x_test_unscaled, _ = dataset_test.get_unscaled_data()
            save_test_inputs_csv(species_root, x_test_unscaled.numpy(), feature_names)

        experiment_info = {
            "scheme": SCHEME,
            "experiment_name": experiment_name,
            "fullrun_timestamp": fullrun_timestamp,
            "train_file": dictionary[SCHEME]["main_dataset"],
            "test_file": dictionary[SCHEME]["main_dataset_test"],
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
            "activation": activation,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "patience": patience,
            "max_epochs": max_epochs,
            "x_train_shape": list(x_train.shape),
            "y_train_shape": list(y_train.shape),
            "overwrite_existing_combination": OVERWRITE_EXISTING_COMBINATION,
            "none_placeholder_note": (
                "NONE is only used for folder/display padding. "
                "The model was trained only with kept_species."
            ),
        }
        save_json(os.path.join(species_root, "experiment_info.json"), experiment_info)

        combination_results = []
        total_combination_runs = len(ARCHITECTURES) * len(SEEDS)
        combination_completed = 0

        for seed in SEEDS:
            for hidden_size in ARCHITECTURES:
                metrics = run_single_configuration(
                    scheme=SCHEME,
                    fullrun_timestamp=fullrun_timestamp,
                    experiment_name=experiment_name,
                    species_root=species_root,
                    dataset_train=dataset_train,
                    dataset_test=dataset_test,
                    kept_species=target_species,
                    display_species_for_folder=display_species,
                    hidden_size=hidden_size,
                    seed=seed,
                    activation=activation,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    verbose_epoch_losses=verbose_epoch_losses,
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
                    f"test_mse={metrics['test_mse']:.3e}"
                )

        save_global_summary(species_root, combination_results, filename="summary.txt")
        save_summary_csv(species_root, combination_results, filename="summary.csv")
        save_seed_aggregates(species_root, combination_results)

        print(f"Finished combination: display={display_species} | kept={target_species}")

    if queue_results:
        save_summary_csv(fullrun_root, queue_results, filename="fullrun_queue_summary.csv")
        save_global_summary(fullrun_root, queue_results, filename="fullrun_queue_summary.txt")
        save_seed_aggregates(fullrun_root, queue_results)

    print("")
    print("Full queue done.")