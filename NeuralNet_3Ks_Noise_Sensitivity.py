from tqdm import tqdm

import copy
import hashlib
import itertools
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.NeuralNetworkModels import NeuralNet
from src.config import dict as dictionary


# ======================================================================================
# SETUP
# ======================================================================================

SCHEME = "O2_novib"
EXPERIMENT_NAME = "Individual_Noise"

BASE_RESULTS_DIR = Path("Results_NN")
SAVED_WEIGHTS_ROOT = Path("saved_weights")

# Main output root requested by you:
# Results_NN/O2_novib/Individual_Noise
INDIVIDUAL_NOISE_RESULTS_ROOT = BASE_RESULTS_DIR / SCHEME / EXPERIMENT_NAME

# --------------------------------------------------------------------------------------
# Species combinations to evaluate.
# The code automatically generates all non-empty noisy subsets inside each combination.
# Example:
#   ["O2(X)", "O2(a)"] -> noisy subsets:
#       O2(X)
#       O2(a)
#       O2(X) + O2(a)
#
# O2(X) /  O2(a)  /  O2(b)  / O2(Hz) / O2+(X) / O(3P)
# O(1D) / O+(gnd) / O-(gnd) /  O3(X) / O3(exc)
# --------------------------------------------------------------------------------------
SPECIES_CONFIGS = [
    ["O2(a)", "O2(b)"],
]

# --------------------------------------------------------------------------------------
# Architectures to evaluate.
# Folder names in saved_weights will be exactly: "30, 30", "30, 30, 30", "50, 50".
# --------------------------------------------------------------------------------------
ARCHITECTURES = [
    (30, 30),
    (30, 30, 30),
    (50, 50),
]

# --------------------------------------------------------------------------------------
# Noise setup.
# 0.005 means 0.5%, 0.01 means 1%, 0.10 means 10%.
#
# IMPORTANT:
#   - 0% is handled as one shared clean baseline per species/architecture/seed.
#   - For plotting, every noisy-subset line starts from this same 0% point.
#   - Nonzero noise levels are applied to each generated noisy subset.
# --------------------------------------------------------------------------------------
NOISE_STDS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
NOISE_REPEATS = 20
NOISE_BASE_SEED = 12345
INCLUDE_CLEAN_BASELINE = True

# Seeds used for NN weights / train-validation split / training shuffle.
SEEDS = list(range(32, 52))

# Model/training setup. Must match previous runs if you want saved_weights reuse.
ACTIVATION = "tanh"
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
MAX_EPOCHS = 5000
PATIENCE = 100
VAL_SPLIT = 0.1
VERBOSE_EPOCH_LOSSES = False

# Noise handling. Matches your current noise robustness workflow, except noise is applied
# only to selected species columns instead of all columns.
RESAMPLE_NEGATIVE_DENSITIES = True
MAX_NOISE_RESAMPLE_ATTEMPTS = 1000

# Append/reuse behavior.
# True  -> skip rows already present in the relevant subset noise_results.csv.
# False -> recompute requested rows, append them, and de-duplicate keeping the newest row.
SKIP_EXISTING_EVALUATIONS = True

# Global files are rebuilt from all valid existing subset folders after each run.
REBUILD_GLOBAL_FILES_FROM_ALL_SUBSET_FOLDERS = True

# --------------------------------------------------------------------------------------
# Plot options.
# One plot is generated per species combination and per architecture.
# Each line is one noisy subset, and every line starts from the shared 0% baseline.
# --------------------------------------------------------------------------------------
SAVE_PLOTS_AFTER_RUN = True
PLOT_MAIN_METRIC = "test_mse_scaled"
PLOT_MAIN_METRIC_LABEL = "Scaled MSE"
PLOT_USE_LOG_Y = True
PLOT_SAVE_PNG = True
PLOT_SAVE_PDF = True
PLOT_DPI = 300
PLOT_MAX_LEGEND_COLUMNS = 2

# Use a clear cycle. The first three match the colors already used in your previous code.
PLOT_COLORS = [
    "green",
    "red",
    "blue",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
PLOT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]

# If using a log y-axis, symmetric std error bars can sometimes extend below zero.
# This caps only the lower error bar for plotting. The CSV values are not modified.
CAP_LOWER_ERRORBAR_ON_LOG_Y = True


# ======================================================================================
# SPECIES MAP AND DATASET
# ======================================================================================

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
CLEAN_BASELINE_ID = "clean_baseline"
CLEAN_BASELINE_LABEL = "Clean baseline"
CLEAN_BASELINE_POSITIONS = "0"
CLEAN_BASELINE_NAMES = "clean_baseline"


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


# ======================================================================================
# GENERAL HELPERS
# ======================================================================================


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


def arch_to_file_token(hidden_size_or_text):
    if isinstance(hidden_size_or_text, (tuple, list)):
        text = arch_to_folder_name(hidden_size_or_text)
    else:
        text = str(hidden_size_or_text)
    return text.replace(",", "_").replace(" ", "").replace("__", "_")


def noise_label(noise_std):
    return f"{100.0 * float(noise_std):g}%"


def stable_int_from_text(text, modulus=1_000_000_000):
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(modulus)


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


def validate_architectures(architectures):
    if not architectures:
        raise ValueError("ARCHITECTURES cannot be empty.")
    for arch in architectures:
        if not isinstance(arch, (tuple, list)) or not arch:
            raise ValueError(f"Invalid architecture: {arch}")
        if any(int(n) <= 0 for n in arch):
            raise ValueError(f"Architecture values must be positive: {arch}")


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


# ======================================================================================
# NOISY SUBSET HELPERS
# ======================================================================================


def noisy_subset_to_id(noisy_positions_1based, noisy_species_names):
    if not noisy_positions_1based:
        return CLEAN_BASELINE_ID
    positions_part = "_".join(str(p) for p in noisy_positions_1based)
    species_part = "__".join(safe_path_token(sp) for sp in noisy_species_names)
    return f"noisy_{positions_part}__{species_part}"


def noisy_subset_to_label(noisy_species_names):
    if not noisy_species_names:
        return CLEAN_BASELINE_LABEL
    return " + ".join(str(sp) for sp in noisy_species_names)


def generate_noisy_subsets(kept_species):
    """Generate all non-empty noisy subsets inside kept_species.

    Positions are human-friendly, 1-based, and refer to the order inside kept_species.
    Example kept_species = ["O2(X)", "O2(a)"]:
        position 1 -> O2(X)
        position 2 -> O2(a)
    """
    validate_species_config(kept_species)

    subsets = []
    n = len(kept_species)
    for subset_size in range(1, n + 1):
        for positions_zero_based in itertools.combinations(range(n), subset_size):
            positions_1based = tuple(p + 1 for p in positions_zero_based)
            names = tuple(kept_species[p] for p in positions_zero_based)
            subset_id = noisy_subset_to_id(positions_1based, names)
            subsets.append(
                {
                    "noisy_subset_id": subset_id,
                    "noisy_subset_label": noisy_subset_to_label(names),
                    "noisy_species_positions": ", ".join(str(p) for p in positions_1based),
                    "noisy_species_names": ", ".join(names),
                    "num_noisy_species": int(len(names)),
                    "noisy_positions_zero_based": positions_zero_based,
                    "noisy_positions_1based": positions_1based,
                    "noisy_species_names_list": names,
                }
            )
    return subsets


def clean_baseline_subset_info():
    return {
        "noisy_subset_id": CLEAN_BASELINE_ID,
        "noisy_subset_label": CLEAN_BASELINE_LABEL,
        "noisy_species_positions": CLEAN_BASELINE_POSITIONS,
        "noisy_species_names": CLEAN_BASELINE_NAMES,
        "num_noisy_species": 0,
        "noisy_positions_zero_based": tuple(),
        "noisy_positions_1based": tuple(),
        "noisy_species_names_list": tuple(),
    }


# ======================================================================================
# SAVED-WEIGHTS CACHE AND DATASET LOADING
# ======================================================================================


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


# ======================================================================================
# TRAINING / EVALUATION
# ======================================================================================


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


# ======================================================================================
# SCALING AND INDIVIDUAL-NOISE HELPERS
# ======================================================================================


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


def selected_noise_column_indices(noisy_positions_zero_based, n_kept, num_pressure_conditions):
    noisy_positions_zero_based = tuple(int(p) for p in noisy_positions_zero_based)
    if not noisy_positions_zero_based:
        return []

    columns = []
    for p in range(num_pressure_conditions):
        pressure_offset = p * n_kept
        for pos in noisy_positions_zero_based:
            if pos < 0 or pos >= n_kept:
                raise ValueError(
                    f"Noisy position {pos + 1} is invalid for a combination with {n_kept} species."
                )
            columns.append(pressure_offset + pos)
    return columns


def make_noisy_inputs_unscaled_selected(
    x_clean_unscaled,
    noise_std,
    rng,
    noisy_positions_zero_based,
    n_kept,
    num_pressure_conditions,
):
    """Apply independent multiplicative Gaussian noise only to selected species columns.

    If species 1 and 2 are selected, species 1 and species 2 receive independent noise
    values for every sample and every pressure condition.
    """
    x_clean_unscaled = np.asarray(x_clean_unscaled, dtype=np.float64)
    noise_std = float(noise_std)

    if noise_std == 0.0 or not noisy_positions_zero_based:
        return x_clean_unscaled.copy()

    selected_cols = selected_noise_column_indices(
        noisy_positions_zero_based=noisy_positions_zero_based,
        n_kept=n_kept,
        num_pressure_conditions=num_pressure_conditions,
    )

    x_noisy = x_clean_unscaled.copy()
    clean_selected = x_clean_unscaled[:, selected_cols]

    multiplicative_noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=clean_selected.shape,
    )
    noisy_selected = clean_selected * (1.0 + multiplicative_noise)

    if RESAMPLE_NEGATIVE_DENSITIES:
        negative_mask = noisy_selected < 0.0
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
            noisy_selected[negative_mask] = clean_selected[negative_mask] * (1.0 + new_noise)
            negative_mask = noisy_selected < 0.0

    x_noisy[:, selected_cols] = noisy_selected
    return x_noisy


def noise_rng_seed(seed, noise_std, noise_repeat, noisy_subset_id):
    # Include the noisy subset in the seed so that different subsets do not accidentally
    # reuse exactly the same random stream.
    subset_component = stable_int_from_text(noisy_subset_id, modulus=1_000_000_000)
    return int(
        int(NOISE_BASE_SEED)
        + int(seed) * 1_000_000
        + int(noise_repeat) * 10_000
        + int(round(float(noise_std) * 1_000_000))
        + subset_component
    )


def run_individual_noise_for_single_model(
    model,
    dataset_test,
    kept_species,
    hidden_size,
    seed,
    noise_std,
    noise_repeat,
    noisy_subset_info,
    training_record,
):
    nspecies = dictionary[SCHEME]["n_densities"]
    num_pressure_conditions = dictionary[SCHEME]["n_conditions"]
    n_kept = len(kept_species)

    x_test_unscaled, _ = dataset_test.get_unscaled_data()
    _, y_test_scaled = dataset_test.get_data()

    rng_seed = noise_rng_seed(
        seed=seed,
        noise_std=noise_std,
        noise_repeat=noise_repeat,
        noisy_subset_id=noisy_subset_info["noisy_subset_id"],
    )
    rng = np.random.default_rng(rng_seed)

    x_noisy_unscaled = make_noisy_inputs_unscaled_selected(
        x_test_unscaled.numpy(),
        noise_std=noise_std,
        rng=rng,
        noisy_positions_zero_based=noisy_subset_info["noisy_positions_zero_based"],
        n_kept=n_kept,
        num_pressure_conditions=num_pressure_conditions,
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
        "noise_percent": float(100.0 * float(noise_std)),
        "noise_label": noise_label(noise_std),
        "noise_rng_seed": int(rng_seed),
        "noisy_subset_id": noisy_subset_info["noisy_subset_id"],
        "noisy_subset_label": noisy_subset_info["noisy_subset_label"],
        "noisy_species_positions": noisy_subset_info["noisy_species_positions"],
        "noisy_species_names": noisy_subset_info["noisy_species_names"],
        "num_noisy_species": int(noisy_subset_info["num_noisy_species"]),
        "reused_saved_weights": bool(training_record.get("reused_saved_weights", False)),
        "saved_weights_path": training_record.get("saved_weights_path", ""),
        **metrics,
    }
    return row


# ======================================================================================
# APPEND-SAFE RESULTS MANAGEMENT
# ======================================================================================


def normalize_noise_std_for_key(value):
    return int(round(float(value) * 1_000_000_000_000))


def result_key_from_values(noisy_subset_id, hidden_size_text, seed, noise_std, noise_repeat):
    return (
        str(noisy_subset_id),
        str(hidden_size_text),
        int(seed),
        normalize_noise_std_for_key(noise_std),
        int(noise_repeat),
    )


def result_key_from_row(row):
    return result_key_from_values(
        row["noisy_subset_id"],
        row["hidden_size"],
        row["seed"],
        row["noise_std"],
        row["noise_repeat"],
    )


def species_results_folder(results_root, kept_species):
    return Path(results_root) / species_config_to_name(kept_species)


def subset_results_folder(results_root, kept_species, noisy_subset_id):
    return species_results_folder(results_root, kept_species) / str(noisy_subset_id)


def subset_noise_results_path(results_root, kept_species, noisy_subset_id):
    return subset_results_folder(results_root, kept_species, noisy_subset_id) / "noise_results.csv"


def load_existing_subset_results(results_root, kept_species, noisy_subset_id):
    path = subset_noise_results_path(results_root, kept_species, noisy_subset_id)
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read existing results at {path}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    required = ["noisy_subset_id", "hidden_size", "seed", "noise_std", "noise_repeat"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(
            f"Warning: existing results at {path} are missing columns {missing}. "
            "They will not be used for skipping, but they will be preserved during append."
        )
    return df


def existing_keys_for_subset(existing_df):
    if existing_df.empty:
        return set()
    required = ["noisy_subset_id", "hidden_size", "seed", "noise_std", "noise_repeat"]
    if any(col not in existing_df.columns for col in required):
        return set()
    return {result_key_from_row(row) for _, row in existing_df.iterrows()}


def deduplicate_results(df):
    if df.empty:
        return df

    required = [
        "scheme",
        "experiment_name",
        "species_config_name",
        "noisy_subset_id",
        "hidden_size",
        "seed",
        "noise_repeat",
        "noise_std",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(
            "Warning: cannot fully de-duplicate because these columns are missing: "
            + ", ".join(missing)
        )
        return df

    df = df.copy()
    df["__noise_std_key"] = df["noise_std"].apply(normalize_noise_std_for_key)
    duplicate_key = [
        "scheme",
        "experiment_name",
        "species_config_name",
        "noisy_subset_id",
        "hidden_size",
        "seed",
        "noise_repeat",
        "__noise_std_key",
    ]
    df = df.drop_duplicates(subset=duplicate_key, keep="last")
    df = df.drop(columns=["__noise_std_key"])
    return df


def sort_individual_noise_results(df):
    if df.empty:
        return df

    preferred_cols = [
        "num_species_kept",
        "species_config_name",
        "num_noisy_species",
        "noisy_subset_id",
        "hidden_size",
        "seed",
        "noise_std",
        "noise_repeat",
    ]
    sort_cols = [col for col in preferred_cols if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def aggregate_individual_noise_results(df):
    if df.empty:
        return df.copy()

    group_cols = [
        "scheme",
        "experiment_name",
        "species_config_name",
        "kept_species",
        "num_species_kept",
        "input_size",
        "output_size",
        "hidden_size",
        "noisy_subset_id",
        "noisy_subset_label",
        "noisy_species_positions",
        "noisy_species_names",
        "num_noisy_species",
        "noise_std",
        "noise_percent",
        "noise_label",
    ]

    missing_group_cols = [col for col in group_cols if col not in df.columns]
    if missing_group_cols:
        raise ValueError(
            "Cannot aggregate individual-noise results. Missing columns: "
            + ", ".join(missing_group_cols)
        )

    metric_cols = [col for col in df.columns if col.endswith("_scaled")]
    if not metric_cols:
        raise ValueError("Cannot aggregate individual-noise results. No '*_scaled' metric columns were found.")

    agg_dict = {col: ["mean", "std", "min", "max"] for col in metric_cols}
    agg_dict["seed"] = ["nunique"]
    agg_dict["noise_repeat"] = ["nunique", "count"]

    work_df = df.copy()
    if "reused_saved_weights" in work_df.columns:
        work_df["reused_saved_weights_int"] = (
            work_df["reused_saved_weights"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
            .astype(int)
        )
        agg_dict["reused_saved_weights_int"] = ["mean", "sum"]

    agg = work_df.groupby(group_cols, as_index=False).agg(agg_dict)
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

    sort_cols = [
        "num_species_kept",
        "species_config_name",
        "hidden_size",
        "num_noisy_species",
        "noisy_subset_id",
        "noise_percent",
    ]
    return agg.sort_values(sort_cols).reset_index(drop=True)


def append_current_results_to_subset_folders(results_root, current_results):
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    current_df = pd.DataFrame(current_results)
    if current_df.empty:
        print("No new evaluations were produced. Existing subset results are preserved.")
        return

    group_cols = ["species_config_name", "noisy_subset_id"]
    for (species_config_name, noisy_subset_id), new_subset_df in current_df.groupby(group_cols, sort=False):
        subset_root = results_root / str(species_config_name) / str(noisy_subset_id)
        subset_root.mkdir(parents=True, exist_ok=True)

        path = subset_root / "noise_results.csv"
        if path.exists():
            try:
                existing_df = pd.read_csv(path)
            except Exception as exc:
                print(f"Warning: could not read {path}; preserving by writing a backup first. Error: {exc}")
                backup_path = subset_root / "noise_results_unreadable_backup.csv"
                path.replace(backup_path)
                existing_df = pd.DataFrame()
        else:
            existing_df = pd.DataFrame()

        combined = pd.concat([existing_df, new_subset_df], ignore_index=True, sort=False)
        combined = deduplicate_results(combined)
        combined = sort_individual_noise_results(combined)

        combined.to_csv(path, index=False)

        subset_agg = aggregate_individual_noise_results(combined)
        subset_agg.to_csv(subset_root / "noise_aggregate_summary.csv", index=False)

        print(
            f"Updated {path}: {len(existing_df)} existing row(s), "
            f"{len(new_subset_df)} new row(s), {len(combined)} stored row(s)."
        )


def is_valid_species_results_folder(folder):
    folder = Path(folder)

    if not folder.is_dir():
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


def is_valid_subset_results_folder(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return False
    if not (folder / "noise_results.csv").exists():
        return False
    if folder.name == CLEAN_BASELINE_ID:
        return True
    return folder.name.startswith("noisy_")


def load_subset_noise_results_if_valid(folder):
    folder = Path(folder)

    if not is_valid_subset_results_folder(folder):
        return None

    noise_results_path = folder / "noise_results.csv"

    try:
        df = pd.read_csv(noise_results_path)
    except Exception as exc:
        print(f"Skipping {folder}: could not read noise_results.csv ({exc})")
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
        "noisy_subset_id",
        "noisy_subset_label",
        "noisy_species_positions",
        "noisy_species_names",
        "num_noisy_species",
        "test_mse_scaled",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Skipping {folder}: noise_results.csv is missing columns: "
            + ", ".join(missing_cols)
        )
        return None

    if df.empty:
        print(f"Skipping {folder}: noise_results.csv is empty.")
        return None

    subset_ids = df["noisy_subset_id"].astype(str).dropna().unique()
    if len(subset_ids) != 1:
        print(f"Skipping {folder}: expected exactly one noisy_subset_id, found {len(subset_ids)}.")
        return None

    subset_id = subset_ids[0]
    if subset_id != folder.name:
        print(
            f"Skipping {folder}: folder name does not match noisy_subset_id "
            f"inside CSV ({subset_id})."
        )
        return None

    return df


def rebuild_global_files_from_subset_folders(results_root):
    """Rebuild fullrun CSV files from first-level species folders and second-level subset folders."""
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    all_subset_dfs = []
    included_subset_folders = []

    for species_folder in sorted(results_root.iterdir(), key=lambda p: p.name):
        if not is_valid_species_results_folder(species_folder):
            continue

        for subset_folder in sorted(species_folder.iterdir(), key=lambda p: p.name):
            subset_df = load_subset_noise_results_if_valid(subset_folder)
            if subset_df is None:
                continue

            all_subset_dfs.append(subset_df)
            included_subset_folders.append(str(subset_folder.relative_to(results_root)))

    if not all_subset_dfs:
        print(
            "No valid individual-noise subset folders found. "
            "Global fullrun files were not rebuilt."
        )
        return pd.DataFrame(), pd.DataFrame()

    full_df = pd.concat(all_subset_dfs, ignore_index=True, sort=False)
    full_df = deduplicate_results(full_df)
    full_df = sort_individual_noise_results(full_df)

    full_df.to_csv(results_root / "fullrun_individual_noise_results.csv", index=False)

    full_agg = aggregate_individual_noise_results(full_df)
    full_agg.to_csv(results_root / "fullrun_individual_noise_aggregate_summary.csv", index=False)

    # Also save species-level combined files for convenience.
    for species_config_name, species_df in full_df.groupby("species_config_name", sort=False):
        species_root = results_root / str(species_config_name)
        species_root.mkdir(parents=True, exist_ok=True)
        species_df = sort_individual_noise_results(species_df)
        species_df.to_csv(species_root / "individual_noise_results.csv", index=False)
        species_agg = aggregate_individual_noise_results(species_df)
        species_agg.to_csv(species_root / "individual_noise_aggregate_summary.csv", index=False)

    rebuild_info = {
        "rebuild_policy": (
            "fullrun files rebuilt from valid first-level species-combination folders "
            "and valid second-level individual-noise subset folders containing noise_results.csv"
        ),
        "included_subset_folders": included_subset_folders,
        "num_included_subset_folders": int(len(included_subset_folders)),
        "num_fullrun_rows": int(len(full_df)),
        "num_fullrun_aggregate_rows": int(len(full_agg)),
    }
    save_json(results_root / "fullrun_rebuild_info.json", rebuild_info)

    print("Rebuilt global individual-noise files from subset folders.")
    print(f"Included {len(included_subset_folders)} subset folder(s).")

    return full_df, full_agg


# ======================================================================================
# TASK PLANNING
# ======================================================================================


def nonzero_noise_stds():
    return [float(s) for s in NOISE_STDS if float(s) != 0.0]


def plan_tasks_for_species(results_root, kept_species):
    validate_species_config(kept_species)

    species_name = species_config_to_name(kept_species)
    tasks_to_run = []
    total_requested = 0
    total_existing = 0

    # Shared clean baseline: one evaluation per architecture/seed, with noise_repeat = 0.
    if INCLUDE_CLEAN_BASELINE:
        baseline_info = clean_baseline_subset_info()
        existing_df = load_existing_subset_results(
            results_root,
            kept_species,
            baseline_info["noisy_subset_id"],
        )
        existing_keys = existing_keys_for_subset(existing_df)

        for hidden_size in ARCHITECTURES:
            hidden_size_text = arch_to_folder_name(hidden_size)
            for seed in SEEDS:
                total_requested += 1
                key = result_key_from_values(
                    baseline_info["noisy_subset_id"],
                    hidden_size_text,
                    seed,
                    0.0,
                    0,
                )
                if SKIP_EXISTING_EVALUATIONS and key in existing_keys:
                    total_existing += 1
                    continue

                tasks_to_run.append(
                    {
                        "species_config_name": species_name,
                        "kept_species": kept_species,
                        "hidden_size": hidden_size,
                        "hidden_size_text": hidden_size_text,
                        "seed": int(seed),
                        "noise_std": 0.0,
                        "noise_repeat": 0,
                        "noisy_subset_info": baseline_info,
                    }
                )

    # Nonzero noise: all non-empty noisy subsets.
    for subset_info in generate_noisy_subsets(kept_species):
        existing_df = load_existing_subset_results(
            results_root,
            kept_species,
            subset_info["noisy_subset_id"],
        )
        existing_keys = existing_keys_for_subset(existing_df)

        for hidden_size in ARCHITECTURES:
            hidden_size_text = arch_to_folder_name(hidden_size)
            for seed in SEEDS:
                for noise_std in nonzero_noise_stds():
                    for noise_repeat in range(NOISE_REPEATS):
                        total_requested += 1
                        key = result_key_from_values(
                            subset_info["noisy_subset_id"],
                            hidden_size_text,
                            seed,
                            noise_std,
                            noise_repeat,
                        )
                        if SKIP_EXISTING_EVALUATIONS and key in existing_keys:
                            total_existing += 1
                            continue

                        tasks_to_run.append(
                            {
                                "species_config_name": species_name,
                                "kept_species": kept_species,
                                "hidden_size": hidden_size,
                                "hidden_size_text": hidden_size_text,
                                "seed": int(seed),
                                "noise_std": float(noise_std),
                                "noise_repeat": int(noise_repeat),
                                "noisy_subset_info": subset_info,
                            }
                        )

    return {
        "species_config_name": species_name,
        "kept_species": kept_species,
        "tasks_to_run": tasks_to_run,
        "total_requested": int(total_requested),
        "total_existing": int(total_existing),
        "total_to_run": int(len(tasks_to_run)),
        "generated_noisy_subsets": [
            {
                "noisy_subset_id": s["noisy_subset_id"],
                "noisy_subset_label": s["noisy_subset_label"],
                "noisy_species_positions": s["noisy_species_positions"],
                "noisy_species_names": s["noisy_species_names"],
                "num_noisy_species": int(s["num_noisy_species"]),
            }
            for s in generate_noisy_subsets(kept_species)
        ],
    }


def plan_all_tasks(results_root):
    task_plan = {}
    total_requested = 0
    total_existing = 0
    total_to_run = 0

    for kept_species in SPECIES_CONFIGS:
        species_plan = plan_tasks_for_species(results_root, kept_species)
        task_plan[species_plan["species_config_name"]] = species_plan
        total_requested += species_plan["total_requested"]
        total_existing += species_plan["total_existing"]
        total_to_run += species_plan["total_to_run"]

    return task_plan, total_requested, total_existing, total_to_run


def save_run_info(results_root, task_plan, total_requested, total_existing, total_to_run):
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    run_info = {
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "results_root": str(results_root),
        "saved_weights_root": str(SAVED_WEIGHTS_ROOT),
        "species_configs": SPECIES_CONFIGS,
        "architectures": [list(a) for a in ARCHITECTURES],
        "seeds": SEEDS,
        "noise_stds": NOISE_STDS,
        "noise_labels": [noise_label(s) for s in NOISE_STDS],
        "nonzero_noise_stds_used_for_noisy_subsets": nonzero_noise_stds(),
        "noise_repeats": int(NOISE_REPEATS),
        "noise_base_seed": int(NOISE_BASE_SEED),
        "include_clean_baseline": bool(INCLUDE_CLEAN_BASELINE),
        "clean_baseline_policy": (
            "one shared 0% clean evaluation per species combination / architecture / seed; "
            "plotted as the common starting point for every noisy-subset line"
        ),
        "main_metric": PLOT_MAIN_METRIC,
        "noise_type": (
            "multiplicative Gaussian noise applied only to generated noisy species subsets; "
            "independent random perturbation per selected species column, sample, and pressure"
        ),
        "resample_negative_densities": bool(RESAMPLE_NEGATIVE_DENSITIES),
        "max_noise_resample_attempts": int(MAX_NOISE_RESAMPLE_ATTEMPTS),
        "cache_policy": "load compatible saved_weights model; otherwise train and save before evaluation",
        "append_policy": (
            "new rows are appended to each subset noise_results.csv; duplicate rows for the same "
            "species/noisy_subset/architecture/seed/noise/repeat are de-duplicated keeping the newest row"
        ),
        "skip_existing_evaluations": bool(SKIP_EXISTING_EVALUATIONS),
        "total_requested_evaluations": int(total_requested),
        "total_existing_requested_evaluations_skipped": int(total_existing),
        "total_new_evaluations_to_run": int(total_to_run),
        "species_task_plan_summary": {
            name: {
                "kept_species": plan["kept_species"],
                "total_requested": int(plan["total_requested"]),
                "total_existing": int(plan["total_existing"]),
                "total_to_run": int(plan["total_to_run"]),
                "generated_noisy_subsets": plan["generated_noisy_subsets"],
            }
            for name, plan in task_plan.items()
        },
    }
    save_json(results_root / "individual_noise_run_info.json", run_info)


# ======================================================================================
# PLOTTING
# ======================================================================================


def plot_style_for(index):
    return {
        "color": PLOT_COLORS[index % len(PLOT_COLORS)],
        "marker": PLOT_MARKERS[index % len(PLOT_MARKERS)],
    }


def save_current_figure(plt, output_base):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    if PLOT_SAVE_PNG:
        plt.savefig(output_base.with_suffix(".png"), dpi=PLOT_DPI, bbox_inches="tight")
    if PLOT_SAVE_PDF:
        plt.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def get_metric_mean_std(df, metric_name):
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"
    if mean_col not in df.columns:
        raise ValueError(f"Missing column {mean_col} in aggregate results.")
    if std_col not in df.columns:
        raise ValueError(f"Missing column {std_col} in aggregate results.")

    y = df[mean_col].to_numpy(dtype=float)
    yerr = df[std_col].to_numpy(dtype=float)
    yerr = np.where(np.isnan(yerr), 0.0, yerr)
    return y, yerr


def make_yerr_for_plot(y, yerr):
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    if not PLOT_USE_LOG_Y or not CAP_LOWER_ERRORBAR_ON_LOG_Y:
        return yerr

    lower = np.minimum(yerr, np.maximum(0.0, y * 0.95))
    upper = yerr
    return np.vstack([lower, upper])


def format_noise_axes(ax):
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel(PLOT_MAIN_METRIC_LABEL)
    if PLOT_USE_LOG_Y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)


def save_individual_noise_sensitivity_plots(results_root=None):
    import matplotlib.pyplot as plt

    results_root = Path(results_root) if results_root is not None else INDIVIDUAL_NOISE_RESULTS_ROOT
    aggregate_path = results_root / "fullrun_individual_noise_aggregate_summary.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(
            f"Could not find aggregate summary at:\n{aggregate_path}\n"
            "Run the individual-noise workflow first."
        )

    df = pd.read_csv(aggregate_path)
    if df.empty:
        print("Aggregate summary is empty. No plots were created.")
        return

    plots_root = results_root / "Plots" / "by_species_config_and_architecture"
    plots_root.mkdir(parents=True, exist_ok=True)

    required_cols = [
        "species_config_name",
        "kept_species",
        "hidden_size",
        "noisy_subset_id",
        "noisy_subset_label",
        "noise_percent",
        f"{PLOT_MAIN_METRIC}_mean",
        f"{PLOT_MAIN_METRIC}_std",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError("Cannot plot. Missing columns: " + ", ".join(missing_cols))

    created_files = []

    for species_config_name, species_df in df.groupby("species_config_name", sort=False):
        species_plot_root = plots_root / str(species_config_name)
        species_plot_root.mkdir(parents=True, exist_ok=True)

        for hidden_size, arch_df in species_df.groupby("hidden_size", sort=False):
            baseline_df = arch_df[
                (arch_df["noisy_subset_id"].astype(str) == CLEAN_BASELINE_ID)
                & (arch_df["noise_percent"].astype(float) == 0.0)
            ].copy()

            if baseline_df.empty:
                print(
                    f"Warning: no clean baseline found for {species_config_name}, "
                    f"architecture {hidden_size}. Plot will start at first nonzero noise point."
                )
                baseline_x = np.array([], dtype=float)
                baseline_y = np.array([], dtype=float)
                baseline_yerr = np.array([], dtype=float)
            else:
                baseline_df = baseline_df.sort_values("noise_percent").head(1)
                baseline_x = np.array([0.0], dtype=float)
                baseline_y, baseline_yerr = get_metric_mean_std(baseline_df, PLOT_MAIN_METRIC)

            subset_df_all = arch_df[arch_df["noisy_subset_id"].astype(str) != CLEAN_BASELINE_ID].copy()
            if subset_df_all.empty:
                print(
                    f"Warning: no noisy-subset data found for {species_config_name}, "
                    f"architecture {hidden_size}."
                )
                continue

            fig, ax = plt.subplots(figsize=(10.5, 7.0))

            subset_ids = list(subset_df_all["noisy_subset_id"].drop_duplicates())
            for i, subset_id in enumerate(subset_ids):
                subset_df = subset_df_all[subset_df_all["noisy_subset_id"] == subset_id].copy()
                subset_df = subset_df.sort_values("noise_percent")

                x_nonzero = subset_df["noise_percent"].to_numpy(dtype=float)
                y_nonzero, yerr_nonzero = get_metric_mean_std(subset_df, PLOT_MAIN_METRIC)

                x = np.concatenate([baseline_x, x_nonzero])
                y = np.concatenate([baseline_y, y_nonzero])
                yerr = np.concatenate([baseline_yerr, yerr_nonzero])

                style = plot_style_for(i)
                label = str(subset_df["noisy_subset_label"].iloc[0])

                ax.errorbar(
                    x,
                    y,
                    yerr=make_yerr_for_plot(y, yerr),
                    marker=style["marker"],
                    color=style["color"],
                    linewidth=1.7,
                    capsize=4,
                    label=label,
                )

            kept_species_label = str(arch_df["kept_species"].iloc[0])
            ax.set_title(
                "Individual species-noise sensitivity\n"
                f"Architecture: {hidden_size} | Input species: {kept_species_label}"
            )

            xticks = sorted(float(x) for x in arch_df["noise_percent"].unique())
            if 0.0 not in xticks:
                xticks = [0.0] + xticks
            ax.set_xticks(xticks)
            format_noise_axes(ax)

            ax.legend(
                title="Noisy species subset",
                fontsize=9,
                title_fontsize=10,
                frameon=False,
                ncol=PLOT_MAX_LEGEND_COLUMNS,
            )
            fig.tight_layout()

            output_base = species_plot_root / f"architecture_{arch_to_file_token(hidden_size)}__individual_noise_sensitivity"
            save_current_figure(plt, output_base)
            if PLOT_SAVE_PNG:
                created_files.append(str(output_base.with_suffix(".png")))
            if PLOT_SAVE_PDF:
                created_files.append(str(output_base.with_suffix(".pdf")))

    manifest = {
        "results_root": str(results_root),
        "plots_root": str(plots_root),
        "main_metric": PLOT_MAIN_METRIC,
        "main_metric_label": PLOT_MAIN_METRIC_LABEL,
        "use_log_y": bool(PLOT_USE_LOG_Y),
        "error_bar_definition": (
            "standard deviation of the aggregate metric values in each group; "
            "for nonzero noise this combines training-seed and noise-repeat variability; "
            "for 0% this is the clean-baseline variability across training seeds"
        ),
        "clean_baseline_policy": "shared 0% point plotted at the start of every noisy-subset line",
        "num_created_files": int(len(created_files)),
        "created_files": created_files,
    }
    save_json(results_root / "Plots" / "plot_manifest.json", manifest)

    print(f"Saved individual-noise sensitivity plots to: {plots_root}")


# ======================================================================================
# MAIN WORKFLOW
# ======================================================================================


def run_individual_noise_sensitivity():
    validate_all_species_configs(SPECIES_CONFIGS)
    validate_architectures(ARCHITECTURES)

    INDIVIDUAL_NOISE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    task_plan, total_requested, total_existing, total_to_run = plan_all_tasks(INDIVIDUAL_NOISE_RESULTS_ROOT)
    save_run_info(
        INDIVIDUAL_NOISE_RESULTS_ROOT,
        task_plan=task_plan,
        total_requested=total_requested,
        total_existing=total_existing,
        total_to_run=total_to_run,
    )

    print("Requested evaluations:", total_requested)
    print("Already existing requested evaluations:", total_existing)
    print("New evaluations to run:", total_to_run)

    for species_name, plan in task_plan.items():
        print(f"\nSpecies combination: {species_name}")
        print("Kept species:", ", ".join(plan["kept_species"]))
        print("Generated noisy subsets:")
        for subset in plan["generated_noisy_subsets"]:
            print(f"  {subset['noisy_subset_id']}: {subset['noisy_species_names']}")

    current_results = []

    if total_to_run > 0:
        with tqdm(total=total_to_run, desc="Individual-noise evaluations") as pbar:
            for species_name, species_plan in task_plan.items():
                kept_species = species_plan["kept_species"]
                tasks_to_run = species_plan["tasks_to_run"]

                if not tasks_to_run:
                    print(f"Skipping {species_name}: all requested rows already exist.")
                    continue

                dataset_train, dataset_test = load_datasets_with_saved_scalers(SCHEME, kept_species)

                tasks_df = pd.DataFrame(tasks_to_run)
                for (seed, hidden_size_text), group in tasks_df.groupby(["seed", "hidden_size_text"], sort=False):
                    hidden_size = tuple(int(x.strip()) for x in str(hidden_size_text).split(",") if x.strip())

                    model, _, _, training_record = get_or_train_model(
                        scheme=SCHEME,
                        kept_species=kept_species,
                        hidden_size=hidden_size,
                        seed=int(seed),
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

                    for _, task in group.iterrows():
                        row = run_individual_noise_for_single_model(
                            model=model,
                            dataset_test=dataset_test,
                            kept_species=kept_species,
                            hidden_size=hidden_size,
                            seed=int(task["seed"]),
                            noise_std=float(task["noise_std"]),
                            noise_repeat=int(task["noise_repeat"]),
                            noisy_subset_info=task["noisy_subset_info"],
                            training_record=training_record,
                        )
                        current_results.append(row)

                        pbar.set_postfix(
                            species=species_name,
                            arch=hidden_size_text,
                            seed=int(task["seed"]),
                            noisy=row["noisy_subset_label"],
                            noise=row["noise_label"],
                            mse=f"{row['test_mse_scaled']:.3e}",
                        )
                        pbar.update(1)
    else:
        print("No new evaluations required. Rebuilding global files from existing subset folders.")

    append_current_results_to_subset_folders(INDIVIDUAL_NOISE_RESULTS_ROOT, current_results)

    if REBUILD_GLOBAL_FILES_FROM_ALL_SUBSET_FOLDERS:
        full_df, full_agg = rebuild_global_files_from_subset_folders(INDIVIDUAL_NOISE_RESULTS_ROOT)
    else:
        current_df = pd.DataFrame(current_results)
        full_df = current_df
        full_agg = aggregate_individual_noise_results(current_df) if not current_df.empty else pd.DataFrame()
        full_df.to_csv(INDIVIDUAL_NOISE_RESULTS_ROOT / "fullrun_individual_noise_results.csv", index=False)
        full_agg.to_csv(INDIVIDUAL_NOISE_RESULTS_ROOT / "fullrun_individual_noise_aggregate_summary.csv", index=False)

    print(f"\nSaved individual-noise results to: {INDIVIDUAL_NOISE_RESULTS_ROOT}")
    print(f"Full raw rows: {len(full_df)}")
    print(f"Full aggregate rows: {len(full_agg)}")

    if SAVE_PLOTS_AFTER_RUN:
        try:
            save_individual_noise_sensitivity_plots(INDIVIDUAL_NOISE_RESULTS_ROOT)
        except Exception as exc:
            print(f"WARNING: Individual-noise plotting failed after results were saved: {exc}")


if __name__ == "__main__":
    run_individual_noise_sensitivity()
