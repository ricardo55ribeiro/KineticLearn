from tqdm import tqdm

import copy
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.NeuralNetworkModels import NeuralNet
from src.config import dict as dictionary


# ======================================================================================
# SETUP
# ======================================================================================

SCHEME = "O2_novib"
BASE_EXPERIMENT_NAME = "Noise_Error_Rankings"
EXPERIMENT_NAME = f"Ensemble_{BASE_EXPERIMENT_NAME}"

BASE_RESULTS_DIR = Path("Results_NN")
SAVED_WEIGHTS_ROOT = Path("saved_weights")

# Main output root used by this script. Existing species folders here are preserved.
# Ensemble results are intentionally stored separately from the individual-seed rankings:
# Results_NN/O2_novib/Ensemble_(Noise_Error_Rankings)
NOISE_RESULTS_ROOT = BASE_RESULTS_DIR / SCHEME / EXPERIMENT_NAME
RANKINGS_DIR_NAME = "Noise_Architecture_Rankings"

# --------------------------------------------------------------------------------------
# Species combinations to evaluate and rank.
# O2(X) /  O2(a)  /  O2(b)  / O2(Hz) / O2+(X) / O(3P)
# O(1D) / O+(gnd) / O-(gnd) /  O3(X) / O3(exc)
# --------------------------------------------------------------------------------------
SPECIES_CONFIGS = [
    ["O2(a)", "O2(b)", "O3(X)"],
    ["O2(X)", "O2(a)", "O3(X)"],
    ["O2(a)", "O3(X)"],
    ["O2(X)", "O2(a)", "O2(b)"],
    ["O2(a)", "O3(X)", "O3(exc)"],
    ["O2(a)", "O(3P)", "O3(X)"],
    ["O2(a)", "O2(b)", "O3(exc)"],
    ["O2(a)", "O(1D)", "O3(X)"],
    ["O2(a)", "O2(b)", "O(3P)"],
    ["O2(a)", "O2(b)"],
    ["O2(a)", "O2(b)", "O(1D)"],
    ["O3(X)", "O3(exc)"],
    ["O2(b)", "O3(exc)"],
    ["O2(a)", "O2(b)", "O+(gnd)"],
    ["O2(b)", "O(1D)", "O3(exc)"],
    ["O2(a)", "O2(b)", "O2+(X)"],
    ["O2(b)", "O3(X)"],
    ["O2(X)", "O3(X)", "O(3P)"],
    ["O(1D)", "O3(X)"],
    ["O2(X)", "O2(a)", "O(1D)"],
    ["O2(X)", "O2(b)", "O(3P)"],
    ["O2(X)", "O2(b)", "O2(Hz)"],
    ["O2(b)", "O(1D)"],
    ["O2(b)", "O(3P)", "O2+(X)"],
    ["O(1D)", "O3(exc)"],
    ["O2(X)", "O2(b)"],
    ["O2(X)", "O2(b)", "O2+(X)"],
    ["O2(a)", "O3(exc)"],
    ["O2(a)", "O(3P)", "O2+(X)"],
    ["O2(X)", "O2(a)", "O2(Hz)"],
    ["O2(X)", "O2(a)", "O(3P)"],
    ["O2(X)", "O(3P)", "O2+(X)"],
    ["O2(X)", "O2(a)", "O2+(X)"],
    ["O2(X)", "O(3P)", "O2(Hz)"],
    ["O2(X)", "O2(a)"],
    ["O2(a)", "O2(b)", "O2(Hz)"],
]

# --------------------------------------------------------------------------------------
# Architectures to evaluate and rank.
# Folder names in saved_weights will be exactly: "30, 30", "30, 30, 30", "50, 50".
# --------------------------------------------------------------------------------------
ARCHITECTURES = [
    (30, 30),
    (30, 30, 30),
    (50, 50),
]

# --------------------------------------------------------------------------------------
# Noise setup.
# Same percentage noise is applied to every selected input-density feature.
# 0.005 means 0.5%, 0.01 means 1%, 0.10 means 10%.
# --------------------------------------------------------------------------------------
NOISE_STDS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
NOISE_REPEATS = 20
NOISE_BASE_SEED = 12345

# Seeds used for NN weights / train-validation split / training shuffle.
# In this ensemble ranking script, these 20 seeds are averaged at the prediction level.
SEEDS = list(range(32, 52))
ENSEMBLE_SEEDS = SEEDS

# Model/training setup. Must match your previous runs if you want saved_weights reuse.
ACTIVATION = "tanh"
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
MAX_EPOCHS = 5000
PATIENCE = 100
VAL_SPLIT = 0.1
VERBOSE_EPOCH_LOSSES = False

# Noise handling. This matches your current noise robustness workflow.
RESAMPLE_NEGATIVE_DENSITIES = True
MAX_NOISE_RESAMPLE_ATTEMPTS = 1000

# Very important append/reuse switches.
# True  -> do not recompute rows already present in the relevant species noise_results.csv.
# False -> recompute requested rows, append them, and de-duplicate keeping the newest row.
SKIP_EXISTING_EVALUATIONS = True

# Existing species folders are preserved. Global fullrun files and ranking files are rebuilt
# from all valid species folders after the current run; this does NOT delete species results.
REBUILD_GLOBAL_FILES_FROM_ALL_SPECIES_FOLDERS = True

# Ranking metric. Lower is better.
RANKING_METRIC_MEAN = "test_mse_scaled_mean"
RANKING_METRIC_STD = "test_mse_scaled_std"

# Save both detailed global ranking files and split folders/files.
SAVE_GLOBAL_RANKING_CSV = True
SAVE_GLOBAL_RANKING_TXT = True
SAVE_SPLIT_RANKING_FILES = True

# Optional scatter plots. These run only after the global aggregate/ranking files are saved.
SAVE_SCATTER_PLOTS_AFTER_RUN = True

# Scatter-plot options.
SCATTER_ROOT_DIR_NAME = "ScatterPlots"
SCATTER_METRIC_MEAN = RANKING_METRIC_MEAN
SCATTER_METRIC_STD = RANKING_METRIC_STD

# Architectures to plot. Use None to plot every architecture found in the aggregate/ranking CSV.
SCATTER_ARCHITECTURES_TO_PLOT = [
    "30, 30",
    "30, 30, 30",
    "50, 50",
]

# Noise percentages to plot on the y-axis. Use None to plot every nonzero noise found.
# 0% is automatically excluded because it is used as the x-axis baseline.
SCATTER_NOISE_PERCENTS_TO_PLOT = None

SCATTER_USE_LOG_AXES = True
SCATTER_SHOW_DIAGONAL_Y_EQUALS_X = True
SCATTER_SAVE_PNG = True
SCATTER_SAVE_PDF = True
SCATTER_DPI = 300

# Identification style for scatter-plot points.
# Each point is labeled with a compact numeric ID and the same figure includes a right-side key.
SCATTER_LABEL_POINTS_WITH_IDS = True
SCATTER_SHOW_RIGHT_SIDE_KEY = True
SCATTER_POINT_LABEL_FONT_SIZE = 6.5
SCATTER_KEY_FONT_SIZE = 6.0
SCATTER_KEY_TITLE_FONT_SIZE = 8.0

# Zoomed inset for the crowded left-side / low-clean-MSE scatter-plot cluster.
# The zoom region is selected automatically from the largest gap in log10(clean MSE).
SCATTER_SHOW_ZOOM_INSET = True
SCATTER_ZOOM_INSET_MIN_POINTS = 4
SCATTER_ZOOM_INSET_MAX_CLUSTER_FRACTION = 0.65
SCATTER_ZOOM_INSET_PAD_FRACTION = 0.14
SCATTER_ZOOM_INSET_POINT_SIZE = 45
SCATTER_ZOOM_INSET_LABEL_FONT_SIZE = 6.0
SCATTER_ZOOM_INSET_TITLE_FONT_SIZE = 7.5

# Save the exact data used for each scatter plot and one global data table.
SCATTER_SAVE_PER_PLOT_DATA_CSV = True
SCATTER_SAVE_GLOBAL_DATA_CSV = True


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


def noise_to_file_token(noise_percent):
    value = float(noise_percent)
    text = f"{value:g}".replace(".", "p").replace("-", "m")
    return f"noise_{text}_percent"


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
# SCALING AND NOISE HELPERS
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


def make_noisy_inputs_unscaled(x_clean_unscaled, noise_std, rng):
    x_clean_unscaled = np.asarray(x_clean_unscaled, dtype=np.float64)

    if float(noise_std) == 0.0:
        return x_clean_unscaled.copy()

    multiplicative_noise = rng.normal(
        loc=0.0,
        scale=float(noise_std),
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
                scale=float(noise_std),
                size=int(np.sum(negative_mask)),
            )
            x_noisy[negative_mask] = x_clean_unscaled[negative_mask] * (1.0 + new_noise)
            negative_mask = x_noisy < 0.0

    return x_noisy


def ensemble_noise_rng_seed(noise_std, noise_repeat):
    """Seed for the artificial input-noise realization.

    This deliberately does not depend on the neural-network seed. In ensemble mode,
    every network in the ensemble must see the same noisy input realization; only the
    trained-network weights differ across seeds.
    """
    return (
        int(NOISE_BASE_SEED)
        + int(noise_repeat) * 10_000
        + int(round(float(noise_std) * 1_000_000))
    )


def predict_ensemble_scaled(models_by_seed, x_scaled_np):
    """Return mean and per-seed scaled predictions for one noisy input matrix."""
    if not models_by_seed:
        raise ValueError("models_by_seed cannot be empty.")

    seeds = sorted(int(seed) for seed in models_by_seed.keys())
    outputs_by_seed = []

    for seed in seeds:
        outputs_by_seed.append(predict_scaled(models_by_seed[seed], x_scaled_np))

    outputs_by_seed = np.stack(outputs_by_seed, axis=0).astype(np.float64)
    outputs_mean = np.mean(outputs_by_seed, axis=0)
    outputs_std = np.std(outputs_by_seed, axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(outputs_mean)

    return outputs_mean, outputs_std, outputs_by_seed, seeds


def compute_individual_seed_metric_summary(targets_scaled, outputs_by_seed):
    """Diagnostic metrics for the individual seeds before ensemble averaging."""
    targets_scaled = np.asarray(targets_scaled, dtype=np.float64)
    outputs_by_seed = np.asarray(outputs_by_seed, dtype=np.float64)

    per_seed_mse = []
    per_seed_rmse = []

    for outputs_scaled in outputs_by_seed:
        squared_errors = (outputs_scaled - targets_scaled) ** 2
        mse = float(np.mean(squared_errors))
        per_seed_mse.append(mse)
        per_seed_rmse.append(float(np.sqrt(mse)))

    per_seed_mse = np.asarray(per_seed_mse, dtype=np.float64)
    per_seed_rmse = np.asarray(per_seed_rmse, dtype=np.float64)

    return {
        "individual_seed_test_mse_scaled_mean": float(np.mean(per_seed_mse)),
        "individual_seed_test_mse_scaled_std": float(np.std(per_seed_mse, ddof=1)) if len(per_seed_mse) > 1 else 0.0,
        "individual_seed_test_mse_scaled_min": float(np.min(per_seed_mse)),
        "individual_seed_test_mse_scaled_max": float(np.max(per_seed_mse)),
        "individual_seed_test_rmse_scaled_mean": float(np.mean(per_seed_rmse)),
        "individual_seed_test_rmse_scaled_std": float(np.std(per_seed_rmse, ddof=1)) if len(per_seed_rmse) > 1 else 0.0,
        "individual_seed_test_rmse_scaled_min": float(np.min(per_seed_rmse)),
        "individual_seed_test_rmse_scaled_max": float(np.max(per_seed_rmse)),
    }


def compute_ensemble_uncertainty_summary(outputs_std):
    """Simple diagnostics for prediction spread across network seeds."""
    outputs_std = np.asarray(outputs_std, dtype=np.float64)

    metrics = {
        "ensemble_prediction_std_scaled_mean": float(np.mean(outputs_std)),
        "ensemble_prediction_std_scaled_max": float(np.max(outputs_std)),
    }

    for i in range(outputs_std.shape[1]):
        metrics[f"k{i + 1}_ensemble_prediction_std_scaled_mean"] = float(np.mean(outputs_std[:, i]))
        metrics[f"k{i + 1}_ensemble_prediction_std_scaled_max"] = float(np.max(outputs_std[:, i]))

    return metrics


def run_noise_for_seed_ensemble(
    models_by_seed,
    training_records_by_seed,
    dataset_test,
    kept_species,
    hidden_size,
    noise_std,
    noise_repeat,
):
    """Evaluate one seed ensemble for one noise level and one noise repeat.

    The prediction is computed as:
        1. make one noisy input realization;
        2. run every NN seed on that same noisy input;
        3. average the scaled K predictions across NN seeds;
        4. compute the MSE of the ensemble-mean prediction.

    Noise repeats are not averaged here. They remain separate rows and are averaged only
    later in the aggregate/ranking tables.
    """
    nspecies = dictionary[SCHEME]["n_densities"]
    num_pressure_conditions = dictionary[SCHEME]["n_conditions"]

    x_test_unscaled, _ = dataset_test.get_unscaled_data()
    _, y_test_scaled = dataset_test.get_data()

    rng_seed = ensemble_noise_rng_seed(noise_std, noise_repeat)
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

    outputs_mean_scaled, outputs_std_scaled, outputs_by_seed_scaled, ensemble_seeds = predict_ensemble_scaled(
        models_by_seed,
        x_noisy_scaled,
    )

    ensemble_metrics = compute_scaled_metrics(y_test_scaled.numpy(), outputs_mean_scaled)
    individual_summary = compute_individual_seed_metric_summary(y_test_scaled.numpy(), outputs_by_seed_scaled)
    uncertainty_summary = compute_ensemble_uncertainty_summary(outputs_std_scaled)

    reused_flags = [
        bool(training_records_by_seed.get(seed, {}).get("reused_saved_weights", False))
        for seed in ensemble_seeds
    ]
    saved_paths = [
        str(training_records_by_seed.get(seed, {}).get("saved_weights_path", ""))
        for seed in ensemble_seeds
    ]

    row = {
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "base_experiment_name": BASE_EXPERIMENT_NAME,
        "evaluation_mode": "seed_ensemble",
        "ensemble_definition": "mean prediction over neural-network seeds; noise repeats kept separate",
        "species_config_name": species_config_to_name(kept_species),
        "kept_species": ", ".join(kept_species),
        "num_species_kept": int(len(kept_species)),
        "input_size": int(x_noisy_scaled.shape[1]),
        "output_size": int(y_test_scaled.shape[1]),
        "hidden_size": arch_to_folder_name(hidden_size),
        "ensemble_seed_values": ";".join(str(seed) for seed in ensemble_seeds),
        "num_ensemble_seeds": int(len(ensemble_seeds)),
        "noise_repeat": int(noise_repeat),
        "noise_std": float(noise_std),
        "noise_percent": float(100.0 * float(noise_std)),
        "noise_label": noise_label(noise_std),
        "noise_rng_seed": int(rng_seed),
        "num_reused_saved_weights": int(sum(reused_flags)),
        "fraction_reused_saved_weights": float(np.mean(reused_flags)) if reused_flags else np.nan,
        "all_saved_weights_reused": bool(all(reused_flags)) if reused_flags else False,
        "saved_weights_paths": "|".join(saved_paths),
        **ensemble_metrics,
        **individual_summary,
        **uncertainty_summary,
    }
    return row


# ======================================================================================
# APPEND-SAFE RESULTS MANAGEMENT
# ======================================================================================


def normalize_noise_std_for_key(value):
    return int(round(float(value) * 1_000_000_000_000))


def result_key_from_values(hidden_size_text, noise_std, noise_repeat):
    return (
        str(hidden_size_text),
        normalize_noise_std_for_key(noise_std),
        int(noise_repeat),
    )


def result_key_from_row(row):
    return result_key_from_values(
        row["hidden_size"],
        row["noise_std"],
        row["noise_repeat"],
    )


def species_results_folder(noise_results_root, kept_species):
    return Path(noise_results_root) / species_config_to_name(kept_species)


def species_noise_results_path(noise_results_root, kept_species):
    return species_results_folder(noise_results_root, kept_species) / "noise_results.csv"


def load_existing_species_results(noise_results_root, kept_species):
    path = species_noise_results_path(noise_results_root, kept_species)
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read existing results at {path}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    required = ["hidden_size", "noise_std", "noise_repeat"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(
            f"Warning: existing ensemble results at {path} are missing columns {missing}. "
            "They will not be used for skipping, but they will be preserved during append."
        )
    return df


def existing_keys_for_species(existing_df):
    if existing_df.empty:
        return set()
    required = ["hidden_size", "noise_std", "noise_repeat"]
    if any(col not in existing_df.columns for col in required):
        return set()
    return {result_key_from_row(row) for _, row in existing_df.iterrows()}


def deduplicate_results(df):
    if df.empty:
        return df

    required = ["scheme", "experiment_name", "species_config_name", "hidden_size", "noise_repeat", "noise_std"]
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
        "hidden_size",
        "noise_repeat",
        "__noise_std_key",
    ]
    df = df.drop_duplicates(subset=duplicate_key, keep="last")
    df = df.drop(columns=["__noise_std_key"])
    return df


def sort_noise_results(df):
    if df.empty:
        return df

    preferred_cols = [
        "num_species_kept",
        "species_config_name",
        "hidden_size",
        "noise_std",
        "noise_repeat",
    ]
    sort_cols = [col for col in preferred_cols if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def aggregate_noise_results(df):
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
        "noise_std",
        "noise_percent",
        "noise_label",
    ]

    missing_group_cols = [col for col in group_cols if col not in df.columns]
    if missing_group_cols:
        raise ValueError("Cannot aggregate ensemble noise results. Missing columns: " + ", ".join(missing_group_cols))

    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    metric_cols = [col for col in df.columns if "_scaled" in col and col in numeric_cols]
    if not metric_cols:
        raise ValueError("Cannot aggregate ensemble noise results. No numeric '*_scaled*' metric columns were found.")

    agg_dict = {col: ["mean", "std", "min", "max"] for col in metric_cols}
    agg_dict["noise_repeat"] = ["nunique", "count"]

    if "num_ensemble_seeds" in df.columns:
        agg_dict["num_ensemble_seeds"] = ["first", "min", "max"]
    if "ensemble_seed_values" in df.columns:
        agg_dict["ensemble_seed_values"] = ["first"]
    if "fraction_reused_saved_weights" in df.columns:
        agg_dict["fraction_reused_saved_weights"] = ["mean", "min", "max"]
    if "num_reused_saved_weights" in df.columns:
        agg_dict["num_reused_saved_weights"] = ["mean", "min", "max"]
    if "all_saved_weights_reused" in df.columns:
        work_df = df.copy()
        work_df["all_saved_weights_reused_int"] = work_df["all_saved_weights_reused"].astype(int)
        agg_dict["all_saved_weights_reused_int"] = ["mean", "sum"]
    else:
        work_df = df.copy()

    agg = work_df.groupby(group_cols, as_index=False).agg(agg_dict)
    agg.columns = [
        col if isinstance(col, str) else "_".join([c for c in col if c])
        for col in agg.columns.to_flat_index()
    ]

    rename_map = {
        "noise_repeat_nunique": "num_noise_repeats",
        "noise_repeat_count": "num_evaluations",
        "num_ensemble_seeds_first": "num_seeds",
        "ensemble_seed_values_first": "ensemble_seed_values",
        "fraction_reused_saved_weights_mean": "fraction_reused_saved_weights",
        "num_reused_saved_weights_mean": "num_reused_saved_weights_mean",
        "all_saved_weights_reused_int_mean": "fraction_all_saved_weights_reused",
        "all_saved_weights_reused_int_sum": "num_noise_repeats_all_saved_weights_reused",
    }
    agg.rename(columns={k: v for k, v in rename_map.items() if k in agg.columns}, inplace=True)

    # Keep explicit ensemble naming while preserving the previous ranking/scatter column "num_seeds".
    if "num_seeds" in agg.columns and "num_ensemble_seeds" not in agg.columns:
        agg["num_ensemble_seeds"] = agg["num_seeds"]

    sort_cols = ["num_species_kept", "species_config_name", "hidden_size", "noise_percent"]
    return agg.sort_values(sort_cols).reset_index(drop=True)


def append_current_results_to_species_folders(noise_results_root, current_results):
    noise_results_root = Path(noise_results_root)
    noise_results_root.mkdir(parents=True, exist_ok=True)

    current_df = pd.DataFrame(current_results)
    if current_df.empty:
        print("No new evaluations were produced. Existing species results are preserved.")
        return

    for species_config_name, new_species_df in current_df.groupby("species_config_name", sort=False):
        species_root = noise_results_root / str(species_config_name)
        species_root.mkdir(parents=True, exist_ok=True)

        path = species_root / "noise_results.csv"
        if path.exists():
            try:
                existing_df = pd.read_csv(path)
            except Exception as exc:
                print(f"Warning: could not read {path}; preserving by writing a backup first. Error: {exc}")
                backup_path = species_root / "noise_results_unreadable_backup.csv"
                path.replace(backup_path)
                existing_df = pd.DataFrame()
        else:
            existing_df = pd.DataFrame()

        combined = pd.concat([existing_df, new_species_df], ignore_index=True, sort=False)
        combined = deduplicate_results(combined)
        combined = sort_noise_results(combined)

        combined.to_csv(path, index=False)

        species_agg = aggregate_noise_results(combined)
        species_agg.to_csv(species_root / "noise_aggregate_summary.csv", index=False)

        print(
            f"Updated {path}: {len(existing_df)} existing row(s), "
            f"{len(new_species_df)} new row(s), {len(combined)} stored row(s)."
        )


def is_valid_direct_species_results_folder(folder):
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
        "num_ensemble_seeds",
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

    df = deduplicate_results(df)
    df = sort_noise_results(df)
    return df


def rebuild_fullrun_noise_files_from_direct_species_folders(noise_results_root):
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
            "Global fullrun files and rankings were not rebuilt."
        )
        return pd.DataFrame(), pd.DataFrame()

    full_df = pd.concat(all_species_dfs, ignore_index=True, sort=False)
    full_df = deduplicate_results(full_df)
    full_df = sort_noise_results(full_df)

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
    print(f"Included {len(included_folders)} species folder(s).")

    return full_df, full_agg


# ======================================================================================
# TASK PLANNING
# ======================================================================================


def all_requested_tasks_for_species(kept_species):
    tasks = []
    species_config_name = species_config_to_name(kept_species)
    for hidden_size in ARCHITECTURES:
        hidden_size_text = arch_to_folder_name(hidden_size)
        for noise_std in NOISE_STDS:
            for noise_repeat in range(NOISE_REPEATS):
                tasks.append(
                    {
                        "species_config_name": species_config_name,
                        "kept_species": kept_species,
                        "hidden_size": tuple(hidden_size),
                        "hidden_size_text": hidden_size_text,
                        "noise_std": float(noise_std),
                        "noise_repeat": int(noise_repeat),
                        "key": result_key_from_values(hidden_size_text, noise_std, noise_repeat),
                    }
                )
    return tasks


def filter_tasks_against_existing(tasks, existing_keys):
    if not SKIP_EXISTING_EVALUATIONS:
        return tasks
    return [task for task in tasks if task["key"] not in existing_keys]


def plan_all_tasks(noise_results_root):
    all_tasks_by_species = {}
    total_requested = 0
    total_existing = 0
    total_to_run = 0

    for kept_species in SPECIES_CONFIGS:
        species_name = species_config_to_name(kept_species)
        existing_df = load_existing_species_results(noise_results_root, kept_species)
        existing_keys = existing_keys_for_species(existing_df)

        requested_tasks = all_requested_tasks_for_species(kept_species)
        tasks_to_run = filter_tasks_against_existing(requested_tasks, existing_keys)

        all_tasks_by_species[species_name] = {
            "kept_species": kept_species,
            "existing_rows": int(len(existing_df)),
            "existing_keys": existing_keys,
            "requested_tasks": requested_tasks,
            "tasks_to_run": tasks_to_run,
        }

        total_requested += len(requested_tasks)
        total_existing += len(requested_tasks) - len(tasks_to_run)
        total_to_run += len(tasks_to_run)

    return all_tasks_by_species, total_requested, total_existing, total_to_run




# ======================================================================================
# RANKING OUTPUTS
# ======================================================================================


def build_noise_architecture_ranking(aggregate_df):
    if aggregate_df.empty:
        return aggregate_df.copy()

    required_cols = [
        "species_config_name",
        "kept_species",
        "num_species_kept",
        "hidden_size",
        "noise_std",
        "noise_percent",
        "noise_label",
        RANKING_METRIC_MEAN,
    ]
    missing = [col for col in required_cols if col not in aggregate_df.columns]
    if missing:
        raise ValueError("Cannot build ranking. Missing columns: " + ", ".join(missing))

    ranking_cols = [
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
        RANKING_METRIC_MEAN,
        RANKING_METRIC_STD,
        "test_mse_scaled_min",
        "test_mse_scaled_max",
        "test_rmse_scaled_mean",
        "test_rmse_scaled_std",
        "num_seeds",
        "num_ensemble_seeds",
        "ensemble_seed_values",
        "num_noise_repeats",
        "num_evaluations",
        "fraction_reused_saved_weights",
        "num_reused_saved_weights_mean",
        "fraction_all_saved_weights_reused",
    ]
    ranking_cols = [col for col in ranking_cols if col in aggregate_df.columns]

    ranking_df = aggregate_df[ranking_cols].copy()
    ranking_df[RANKING_METRIC_MEAN] = pd.to_numeric(ranking_df[RANKING_METRIC_MEAN], errors="coerce")
    if RANKING_METRIC_STD in ranking_df.columns:
        ranking_df[RANKING_METRIC_STD] = pd.to_numeric(ranking_df[RANKING_METRIC_STD], errors="coerce")

    ranking_df = ranking_df.dropna(subset=[RANKING_METRIC_MEAN]).copy()

    ranking_df = ranking_df.sort_values(
        [
            "noise_percent",
            "hidden_size",
            RANKING_METRIC_MEAN,
            RANKING_METRIC_STD if RANKING_METRIC_STD in ranking_df.columns else RANKING_METRIC_MEAN,
            "species_config_name",
        ]
    ).reset_index(drop=True)

    ranking_df.insert(
        0,
        "rank",
        ranking_df.groupby(["noise_percent", "hidden_size"], sort=False).cumcount() + 1,
    )

    # More convenient reading order.
    first_cols = [
        "noise_percent",
        "noise_label",
        "hidden_size",
        "rank",
        "species_config_name",
        "kept_species",
        RANKING_METRIC_MEAN,
    ]
    if RANKING_METRIC_STD in ranking_df.columns:
        first_cols.append(RANKING_METRIC_STD)
    first_cols = [col for col in first_cols if col in ranking_df.columns]
    remaining_cols = [col for col in ranking_df.columns if col not in first_cols]
    ranking_df = ranking_df[first_cols + remaining_cols]

    return ranking_df


def format_metric_with_std(row):
    mean_value = row.get(RANKING_METRIC_MEAN, np.nan)
    std_value = row.get(RANKING_METRIC_STD, np.nan)
    if pd.isna(std_value):
        return f"{float(mean_value):.6e}"
    return f"{float(mean_value):.6e} ± {float(std_value):.6e}"


def write_ranking_txt_by_noise_then_architecture(path, ranking_df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("Seed-ensemble ranking of species combinations by noise percentage and architecture")
    lines.append("Sorted from best to worst within each block. Lower test_mse_scaled_mean is better.")
    lines.append("Metric is computed after averaging predictions over the 20 neural-network seeds.")
    lines.append("")

    if ranking_df.empty:
        lines.append("No ranking rows available.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    for noise_percent in sorted(ranking_df["noise_percent"].unique()):
        df_noise = ranking_df[ranking_df["noise_percent"] == noise_percent].copy()
        noise_label_text = str(df_noise["noise_label"].iloc[0])

        lines.append("#" * 120)
        lines.append(f"Noise level: {noise_label_text}")
        lines.append("#" * 120)
        lines.append("")

        for architecture in ARCHITECTURES:
            arch_text = arch_to_folder_name(architecture)
            df_block = df_noise[df_noise["hidden_size"].astype(str) == arch_text].copy()
            if df_block.empty:
                continue

            lines.extend(format_single_ranking_block(df_block, f"Architecture: {arch_text}"))
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_ranking_txt_by_architecture_then_noise(path, ranking_df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("Seed-ensemble ranking of species combinations by architecture and noise percentage")
    lines.append("Sorted from best to worst within each block. Lower test_mse_scaled_mean is better.")
    lines.append("Metric is computed after averaging predictions over the 20 neural-network seeds.")
    lines.append("")

    if ranking_df.empty:
        lines.append("No ranking rows available.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    for architecture in ARCHITECTURES:
        arch_text = arch_to_folder_name(architecture)
        df_arch = ranking_df[ranking_df["hidden_size"].astype(str) == arch_text].copy()
        if df_arch.empty:
            continue

        lines.append("#" * 120)
        lines.append(f"Architecture: {arch_text}")
        lines.append("#" * 120)
        lines.append("")

        for noise_percent in sorted(df_arch["noise_percent"].unique()):
            df_block = df_arch[df_arch["noise_percent"] == noise_percent].copy()
            if df_block.empty:
                continue
            noise_label_text = str(df_block["noise_label"].iloc[0])
            lines.extend(format_single_ranking_block(df_block, f"Noise level: {noise_label_text}"))
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def format_single_ranking_block(df_block, title):
    df_block = df_block.sort_values("rank")

    header_rank = "Rank"
    header_combo = "Species combination"
    header_mse = "test_mse_scaled_mean ± std"
    header_n = "Evaluations"
    header_seeds = "Seeds"

    width_rank = max(len(header_rank), len(str(len(df_block)))) + 2
    width_combo = max(len(header_combo), df_block["kept_species"].astype(str).map(len).max()) + 4
    width_mse = max(len(header_mse), 32) + 2
    width_n = max(len(header_n), 11) + 2
    width_seeds = max(len(header_seeds), 6) + 2

    total_width = width_rank + width_combo + width_mse + width_n + width_seeds

    lines = []
    lines.append("=" * total_width)
    lines.append(title)
    lines.append("=" * total_width)
    lines.append(
        f"{header_rank:<{width_rank}}"
        f"{header_combo:<{width_combo}}"
        f"{header_mse:<{width_mse}}"
        f"{header_n:<{width_n}}"
        f"{header_seeds:<{width_seeds}}"
    )
    lines.append("-" * total_width)

    for _, row in df_block.iterrows():
        rank_text = str(int(row["rank"]))
        species_text = str(row["kept_species"])
        mse_text = format_metric_with_std(row)
        evals_text = "" if "num_evaluations" not in row or pd.isna(row["num_evaluations"]) else str(int(row["num_evaluations"]))
        seeds_text = "" if "num_seeds" not in row or pd.isna(row["num_seeds"]) else str(int(row["num_seeds"]))

        lines.append(
            f"{rank_text:<{width_rank}}"
            f"{species_text:<{width_combo}}"
            f"{mse_text:<{width_mse}}"
            f"{evals_text:<{width_n}}"
            f"{seeds_text:<{width_seeds}}"
        )

    return lines


def save_split_ranking_files(rankings_root, ranking_df):
    rankings_root = Path(rankings_root)

    by_noise_root = rankings_root / "by_noise"
    by_arch_root = rankings_root / "by_architecture"
    by_noise_root.mkdir(parents=True, exist_ok=True)
    by_arch_root.mkdir(parents=True, exist_ok=True)

    # noise -> architecture files
    for noise_percent in sorted(ranking_df["noise_percent"].unique()):
        df_noise = ranking_df[ranking_df["noise_percent"] == noise_percent].copy()
        noise_dir = by_noise_root / noise_to_file_token(noise_percent)
        noise_dir.mkdir(parents=True, exist_ok=True)

        for architecture in ARCHITECTURES:
            arch_text = arch_to_folder_name(architecture)
            df_block = df_noise[df_noise["hidden_size"].astype(str) == arch_text].copy()
            if df_block.empty:
                continue

            arch_token = arch_to_file_token(architecture)
            csv_path = noise_dir / f"architecture_{arch_token}.csv"
            txt_path = noise_dir / f"architecture_{arch_token}.txt"

            df_block.to_csv(csv_path, index=False)
            txt_path.write_text(
                "\n".join(format_single_ranking_block(df_block, f"Noise: {df_block['noise_label'].iloc[0]} | Architecture: {arch_text}")),
                encoding="utf-8",
            )

    # architecture -> noise files
    for architecture in ARCHITECTURES:
        arch_text = arch_to_folder_name(architecture)
        df_arch = ranking_df[ranking_df["hidden_size"].astype(str) == arch_text].copy()
        if df_arch.empty:
            continue

        arch_dir = by_arch_root / f"architecture_{arch_to_file_token(architecture)}"
        arch_dir.mkdir(parents=True, exist_ok=True)

        for noise_percent in sorted(df_arch["noise_percent"].unique()):
            df_block = df_arch[df_arch["noise_percent"] == noise_percent].copy()
            if df_block.empty:
                continue

            noise_token = noise_to_file_token(noise_percent)
            csv_path = arch_dir / f"{noise_token}.csv"
            txt_path = arch_dir / f"{noise_token}.txt"

            df_block.to_csv(csv_path, index=False)
            txt_path.write_text(
                "\n".join(format_single_ranking_block(df_block, f"Architecture: {arch_text} | Noise: {df_block['noise_label'].iloc[0]}")),
                encoding="utf-8",
            )


def save_ranking_outputs(noise_results_root, aggregate_df):
    noise_results_root = Path(noise_results_root)
    rankings_root = noise_results_root / RANKINGS_DIR_NAME
    rankings_root.mkdir(parents=True, exist_ok=True)

    ranking_df = build_noise_architecture_ranking(aggregate_df)

    if SAVE_GLOBAL_RANKING_CSV:
        ranking_df.to_csv(rankings_root / "noise_architecture_species_ranking.csv", index=False)

    if SAVE_GLOBAL_RANKING_TXT:
        write_ranking_txt_by_noise_then_architecture(
            rankings_root / "noise_architecture_species_ranking__by_noise_then_architecture.txt",
            ranking_df,
        )
        write_ranking_txt_by_architecture_then_noise(
            rankings_root / "noise_architecture_species_ranking__by_architecture_then_noise.txt",
            ranking_df,
        )

    if SAVE_SPLIT_RANKING_FILES and not ranking_df.empty:
        save_split_ranking_files(rankings_root, ranking_df)

    build_info = {
        "ranking_metric": RANKING_METRIC_MEAN,
        "ranking_direction": "ascending; lower is better",
        "evaluation_mode": "seed_ensemble",
        "ensemble_definition": "mean prediction over neural-network seeds before computing MSE",
        "rank_blocks": "one independent ranking per noise_percent and hidden_size",
        "global_ranking_csv": str(rankings_root / "noise_architecture_species_ranking.csv"),
        "by_noise_txt": str(rankings_root / "noise_architecture_species_ranking__by_noise_then_architecture.txt"),
        "by_architecture_txt": str(rankings_root / "noise_architecture_species_ranking__by_architecture_then_noise.txt"),
        "split_folders": ["by_noise", "by_architecture"] if SAVE_SPLIT_RANKING_FILES else [],
        "num_ranking_rows": int(len(ranking_df)),
        "num_species_configurations_ranked": int(ranking_df["species_config_name"].nunique()) if not ranking_df.empty else 0,
        "num_architectures_ranked": int(ranking_df["hidden_size"].nunique()) if not ranking_df.empty else 0,
        "noise_percent_values_ranked": sorted(float(x) for x in ranking_df["noise_percent"].unique()) if not ranking_df.empty else [],
    }
    save_json(rankings_root / "ranking_build_info.json", build_info)

    txt_lines = [
        "Seed-ensemble noise/architecture ranking build information",
        "",
        f"Ranking metric: {RANKING_METRIC_MEAN}",
        "Metric definition: MSE after averaging scaled K predictions over neural-network seeds",
        "Ranking direction: ascending; lower is better",
        "Rank blocks: one independent ranking per noise_percent and hidden_size",
        f"Number of ranking rows: {len(ranking_df)}",
        f"Ranking root: {rankings_root}",
    ]
    (rankings_root / "ranking_build_info.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"Saved ranking outputs to: {rankings_root}")
    return ranking_df


# ======================================================================================
# SCATTER-PLOT OUTPUTS
# ======================================================================================


def scatter_aggregate_csv_path(noise_results_root):
    return Path(noise_results_root) / "fullrun_noise_aggregate_summary.csv"


def scatter_ranking_csv_path(noise_results_root):
    return Path(noise_results_root) / RANKINGS_DIR_NAME / "noise_architecture_species_ranking.csv"


def scatter_root_path(noise_results_root):
    return Path(noise_results_root) / SCATTER_ROOT_DIR_NAME


def load_scatter_input_table(noise_results_root):
    """Load the aggregate table, with a fallback to the ranking table."""
    aggregate_path = scatter_aggregate_csv_path(noise_results_root)
    ranking_path = scatter_ranking_csv_path(noise_results_root)

    if aggregate_path.exists():
        path = aggregate_path
    elif ranking_path.exists():
        path = ranking_path
    else:
        raise FileNotFoundError(
            "Could not find either input CSV:\n"
            f"  {aggregate_path}\n"
            f"  {ranking_path}\n\n"
            "Run the noise ranking workflow first, then create scatter plots."
        )

    df = pd.read_csv(path)
    print(f"Read scatter-plot input table from:\n{path}")
    return df, str(path)


def require_scatter_columns(df, columns, source_label):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Scatter input table is missing required columns: {', '.join(missing)}\n"
            f"Input table: {source_label}\n"
            f"Available columns: {list(df.columns)}"
        )


def format_scatter_noise_percent(value):
    value = float(value)
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:g}%"


def normalize_scatter_input_table(df, source_label):
    required = [
        "species_config_name",
        "kept_species",
        "hidden_size",
        "noise_percent",
        SCATTER_METRIC_MEAN,
    ]
    require_scatter_columns(df, required, source_label)

    df = df.copy()
    df["species_config_name"] = df["species_config_name"].astype(str)
    df["kept_species"] = df["kept_species"].astype(str)
    df["hidden_size"] = df["hidden_size"].astype(str).str.strip()
    df["noise_percent"] = pd.to_numeric(df["noise_percent"], errors="coerce")
    df[SCATTER_METRIC_MEAN] = pd.to_numeric(df[SCATTER_METRIC_MEAN], errors="coerce")

    if SCATTER_METRIC_STD in df.columns:
        df[SCATTER_METRIC_STD] = pd.to_numeric(df[SCATTER_METRIC_STD], errors="coerce")
    else:
        df[SCATTER_METRIC_STD] = np.nan

    # Reformat from numeric value to guarantee clean labels like "0.5%", "1%", "10%".
    df["noise_label"] = df["noise_percent"].apply(format_scatter_noise_percent)

    # Drop unusable rows.
    df = df.dropna(subset=["noise_percent", SCATTER_METRIC_MEAN]).copy()

    # The expected input file should already be unique, but this makes the plotter safer.
    dedup_key = ["species_config_name", "hidden_size", "noise_percent"]
    df = df.drop_duplicates(subset=dedup_key, keep="last").reset_index(drop=True)

    return df


def scatter_architecture_to_file_token(architecture_text):
    # "30, 30" -> "30_30"; "30, 30, 30" -> "30_30_30".
    text = str(architecture_text).strip()
    return text.replace(",", "_").replace(" ", "").replace("__", "_")


def scatter_base_filename(noise_percent, architecture_text):
    # Exactly this style, e.g. "0.5%__50_50".
    # Do not use Path.with_suffix() later, because "0.5%" contains a dot.
    noise_text = format_scatter_noise_percent(noise_percent)
    arch_text = scatter_architecture_to_file_token(architecture_text)
    return f"{noise_text}__{arch_text}"


def choose_scatter_architectures(df):
    found = list(dict.fromkeys(df["hidden_size"].astype(str).tolist()))

    if SCATTER_ARCHITECTURES_TO_PLOT is None:
        return found

    requested = [str(a).strip() for a in SCATTER_ARCHITECTURES_TO_PLOT]
    missing = [arch for arch in requested if arch not in found]
    if missing:
        print("Warning: these requested scatter-plot architectures were not found in the table:")
        for arch in missing:
            print(f"  {arch}")

    return [arch for arch in requested if arch in found]


def choose_scatter_noise_percents(df):
    found = sorted(float(x) for x in df["noise_percent"].dropna().unique())
    found_nonzero = [x for x in found if not np.isclose(x, 0.0)]

    if SCATTER_NOISE_PERCENTS_TO_PLOT is None:
        return found_nonzero

    requested = [float(x) for x in SCATTER_NOISE_PERCENTS_TO_PLOT if not np.isclose(float(x), 0.0)]
    selected = []
    missing = []

    for wanted in requested:
        matches = [x for x in found_nonzero if np.isclose(x, wanted, rtol=0.0, atol=1e-9)]
        if matches:
            selected.append(matches[0])
        else:
            missing.append(wanted)

    if missing:
        print("Warning: these requested nonzero scatter-plot noise percentages were not found in the table:")
        for noise in missing:
            print(f"  {format_scatter_noise_percent(noise)}")

    # Preserve user order while removing duplicates.
    unique_selected = []
    for x in selected:
        if not any(np.isclose(x, y, rtol=0.0, atol=1e-9) for y in unique_selected):
            unique_selected.append(x)

    return unique_selected


def select_scatter_noise_rows(df, architecture_text, noise_percent):
    return df[
        (df["hidden_size"].astype(str) == str(architecture_text))
        & np.isclose(df["noise_percent"].astype(float), float(noise_percent), rtol=0.0, atol=1e-9)
    ].copy()


def build_scatter_dataframe(df, architecture_text, noise_percent):
    clean_df = select_scatter_noise_rows(df, architecture_text, 0.0)
    noisy_df = select_scatter_noise_rows(df, architecture_text, noise_percent)

    if clean_df.empty:
        print(f"Skipping {format_scatter_noise_percent(noise_percent)} / {architecture_text}: no 0% baseline rows.")
        return pd.DataFrame()
    if noisy_df.empty:
        print(f"Skipping {format_scatter_noise_percent(noise_percent)} / {architecture_text}: no noisy rows.")
        return pd.DataFrame()

    clean_cols = ["species_config_name", "kept_species", SCATTER_METRIC_MEAN, SCATTER_METRIC_STD]
    extra_clean_cols = [
        col
        for col in ["num_species_kept", "input_size", "output_size", "num_seeds", "num_noise_repeats", "num_evaluations"]
        if col in clean_df.columns
    ]
    clean_cols = clean_cols[:2] + extra_clean_cols + clean_cols[2:]

    noisy_cols = ["species_config_name", SCATTER_METRIC_MEAN, SCATTER_METRIC_STD]
    extra_noisy_cols = [
        col
        for col in ["num_seeds", "num_noise_repeats", "num_evaluations"]
        if col in noisy_df.columns
    ]
    noisy_cols = noisy_cols[:1] + extra_noisy_cols + noisy_cols[1:]

    clean_small = clean_df[clean_cols].rename(
        columns={
            SCATTER_METRIC_MEAN: "clean_mse_scaled_mean",
            SCATTER_METRIC_STD: "clean_mse_scaled_std",
            "num_seeds": "clean_num_seeds",
            "num_noise_repeats": "clean_num_noise_repeats",
            "num_evaluations": "clean_num_evaluations",
        }
    )
    noisy_small = noisy_df[noisy_cols].rename(
        columns={
            SCATTER_METRIC_MEAN: "noisy_mse_scaled_mean",
            SCATTER_METRIC_STD: "noisy_mse_scaled_std",
            "num_seeds": "noisy_num_seeds",
            "num_noise_repeats": "noisy_num_noise_repeats",
            "num_evaluations": "noisy_num_evaluations",
        }
    )

    merged = clean_small.merge(noisy_small, on="species_config_name", how="inner")

    if merged.empty:
        print(
            f"Skipping {format_scatter_noise_percent(noise_percent)} / {architecture_text}: "
            "no species combinations are shared between 0% and noisy rows."
        )
        return pd.DataFrame()

    merged.insert(0, "noise_percent", float(noise_percent))
    merged.insert(1, "noise_label", format_scatter_noise_percent(noise_percent))
    merged.insert(2, "hidden_size", str(architecture_text))

    merged["noise_to_clean_mse_ratio"] = (
        merged["noisy_mse_scaled_mean"] / merged["clean_mse_scaled_mean"]
    )
    merged["noise_minus_clean_mse"] = (
        merged["noisy_mse_scaled_mean"] - merged["clean_mse_scaled_mean"]
    )

    merged = merged.sort_values(
        ["noisy_mse_scaled_mean", "clean_mse_scaled_mean", "species_config_name"]
    ).reset_index(drop=True)

    merged.insert(3, "noisy_rank", np.arange(1, len(merged) + 1, dtype=int))
    point_id_width = max(2, len(str(len(merged))))
    merged.insert(4, "point_id", [f"{i:0{point_id_width}d}" for i in range(1, len(merged) + 1)])
    return merged


def finite_positive_scatter_xy(scatter_df):
    x = scatter_df["clean_mse_scaled_mean"].to_numpy(dtype=float)
    y = scatter_df["noisy_mse_scaled_mean"].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if SCATTER_USE_LOG_AXES:
        mask = mask & (x > 0.0) & (y > 0.0)

    return scatter_df.loc[mask].copy(), x[mask], y[mask]


def get_scatter_axis_limits(x, y):
    values = np.concatenate([x, y])
    values = values[np.isfinite(values)]

    if SCATTER_USE_LOG_AXES:
        values = values[values > 0.0]

    if len(values) == 0:
        return None, None

    vmin = float(values.min())
    vmax = float(values.max())

    if np.isclose(vmin, vmax):
        if SCATTER_USE_LOG_AXES:
            lower = vmin / 2.0
            upper = vmax * 2.0
        else:
            margin = abs(vmin) * 0.1 if vmin != 0 else 1.0
            lower = vmin - margin
            upper = vmax + margin
    else:
        if SCATTER_USE_LOG_AXES:
            lower = vmin / 1.35
            upper = vmax * 1.35
        else:
            margin = 0.08 * (vmax - vmin)
            lower = vmin - margin
            upper = vmax + margin

    return lower, upper


def get_padded_scatter_limits(values, use_log_scale, pad_fraction):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if use_log_scale:
        values = values[values > 0.0]

    if len(values) == 0:
        return None

    vmin = float(values.min())
    vmax = float(values.max())

    if use_log_scale:
        log_values = np.log10(values)
        log_min = float(log_values.min())
        log_max = float(log_values.max())

        if np.isclose(log_min, log_max):
            half_range = 0.15
            return 10.0 ** (log_min - half_range), 10.0 ** (log_max + half_range)

        pad = max(float(pad_fraction) * (log_max - log_min), 0.04)
        return 10.0 ** (log_min - pad), 10.0 ** (log_max + pad)

    if np.isclose(vmin, vmax):
        margin = abs(vmin) * 0.15 if vmin != 0.0 else 1.0
    else:
        margin = float(pad_fraction) * (vmax - vmin)

    return vmin - margin, vmax + margin


def choose_scatter_zoom_cluster(plot_df):
    """Automatically select the crowded low-clean-MSE cluster for the inset."""
    if not SCATTER_SHOW_ZOOM_INSET:
        return pd.DataFrame(), None, None

    if len(plot_df) < SCATTER_ZOOM_INSET_MIN_POINTS + 1:
        return pd.DataFrame(), None, None

    x = plot_df["clean_mse_scaled_mean"].to_numpy(dtype=float)
    y = plot_df["noisy_mse_scaled_mean"].to_numpy(dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)

    if SCATTER_USE_LOG_AXES:
        finite_mask = finite_mask & (x > 0.0) & (y > 0.0)

    work_df = plot_df.loc[finite_mask].copy()
    if len(work_df) < SCATTER_ZOOM_INSET_MIN_POINTS + 1:
        return pd.DataFrame(), None, None

    x_work = work_df["clean_mse_scaled_mean"].to_numpy(dtype=float)

    if SCATTER_USE_LOG_AXES:
        x_sort_values = np.log10(x_work)
    else:
        x_sort_values = x_work.copy()

    sort_order = np.argsort(x_sort_values)
    sorted_values = x_sort_values[sort_order]
    gaps = np.diff(sorted_values)

    if len(gaps) == 0 or not np.any(np.isfinite(gaps)):
        return pd.DataFrame(), None, None

    n_points = len(work_df)
    min_cluster_points = max(2, int(SCATTER_ZOOM_INSET_MIN_POINTS))
    max_cluster_points = int(np.floor(float(SCATTER_ZOOM_INSET_MAX_CLUSTER_FRACTION) * n_points))
    max_cluster_points = max(min_cluster_points, min(max_cluster_points, n_points - 1))

    candidate_splits = np.arange(1, len(gaps) + 1, dtype=int)
    valid_mask = (candidate_splits >= min_cluster_points) & (candidate_splits <= max_cluster_points)
    valid_splits = candidate_splits[valid_mask]

    if len(valid_splits) == 0:
        return pd.DataFrame(), None, None

    valid_gaps = gaps[valid_splits - 1]
    finite_gap_mask = np.isfinite(valid_gaps) & (valid_gaps > 0.0)
    if not np.any(finite_gap_mask):
        return pd.DataFrame(), None, None

    valid_splits = valid_splits[finite_gap_mask]
    valid_gaps = valid_gaps[finite_gap_mask]
    best_split = int(valid_splits[int(np.argmax(valid_gaps))])

    cluster_indices = work_df.index.to_numpy()[sort_order[:best_split]]
    zoom_df = plot_df.loc[cluster_indices].copy()

    if len(zoom_df) < SCATTER_ZOOM_INSET_MIN_POINTS:
        return pd.DataFrame(), None, None

    x_limits = get_padded_scatter_limits(
        zoom_df["clean_mse_scaled_mean"].to_numpy(dtype=float),
        SCATTER_USE_LOG_AXES,
        SCATTER_ZOOM_INSET_PAD_FRACTION,
    )
    y_limits = get_padded_scatter_limits(
        zoom_df["noisy_mse_scaled_mean"].to_numpy(dtype=float),
        SCATTER_USE_LOG_AXES,
        SCATTER_ZOOM_INSET_PAD_FRACTION,
    )

    if x_limits is None or y_limits is None:
        return pd.DataFrame(), None, None

    return zoom_df.sort_values("noisy_rank"), x_limits, y_limits


def choose_scatter_inset_axes_bounds(ax, plot_df):
    """Place the inset where it covers as few main-plot points as possible."""
    candidates = [
        (0.56, 0.07, 0.40, 0.36),  # preferred: bottom-right, usually below y = x
        (0.56, 0.55, 0.40, 0.36),
        (0.08, 0.07, 0.40, 0.36),
        (0.08, 0.55, 0.40, 0.36),
        (0.30, 0.07, 0.40, 0.36),
    ]

    x = plot_df["clean_mse_scaled_mean"].to_numpy(dtype=float)
    y = plot_df["noisy_mse_scaled_mean"].to_numpy(dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)

    if not np.any(finite_mask):
        return candidates[0]

    points_display = ax.transData.transform(np.column_stack([x[finite_mask], y[finite_mask]]))
    points_axes = ax.transAxes.inverted().transform(points_display)
    px = points_axes[:, 0]
    py = points_axes[:, 1]

    best_candidate = candidates[0]
    best_score = None
    margin = 0.025

    for candidate in candidates:
        left, bottom, width, height = candidate
        inside = (
            (px >= left - margin)
            & (px <= left + width + margin)
            & (py >= bottom - margin)
            & (py <= bottom + height + margin)
        )
        score = int(np.sum(inside))
        if best_score is None or score < best_score:
            best_score = score
            best_candidate = candidate
            if score == 0:
                break

    return best_candidate


def add_scatter_zoom_inset(ax, plot_df):
    zoom_df, x_limits, y_limits = choose_scatter_zoom_cluster(plot_df)
    if zoom_df.empty:
        return

    inset_bounds = choose_scatter_inset_axes_bounds(ax, plot_df)
    axins = ax.inset_axes(inset_bounds)

    axins.scatter(
        plot_df["clean_mse_scaled_mean"],
        plot_df["noisy_mse_scaled_mean"],
        s=SCATTER_ZOOM_INSET_POINT_SIZE,
        alpha=0.85,
    )

    if SCATTER_SHOW_DIAGONAL_Y_EQUALS_X:
        diagonal_lower = min(x_limits[0], y_limits[0])
        diagonal_upper = max(x_limits[1], y_limits[1])
        axins.plot(
            [diagonal_lower, diagonal_upper],
            [diagonal_lower, diagonal_upper],
            linestyle="--",
            linewidth=0.9,
            label="_nolegend_",
        )

    if SCATTER_USE_LOG_AXES:
        axins.set_xscale("log")
        axins.set_yscale("log")

    axins.set_xlim(*x_limits)
    axins.set_ylim(*y_limits)
    axins.grid(True, which="both", alpha=0.25)
    axins.set_title("Zoom", fontsize=SCATTER_ZOOM_INSET_TITLE_FONT_SIZE)
    axins.tick_params(axis="both", which="major", labelsize=6)
    axins.tick_params(axis="both", which="minor", labelsize=0)

    if SCATTER_LABEL_POINTS_WITH_IDS:
        add_scatter_point_id_labels(
            axins,
            zoom_df,
            font_size=SCATTER_ZOOM_INSET_LABEL_FONT_SIZE,
        )

    # Draw the zoomed region on the main plot without adding an extra legend entry.
    ax.plot(
        [x_limits[0], x_limits[1], x_limits[1], x_limits[0], x_limits[0]],
        [y_limits[0], y_limits[0], y_limits[1], y_limits[1], y_limits[0]],
        linestyle="-",
        linewidth=1.0,
        alpha=0.85,
        color="0.25",
        label="_nolegend_",
    )

def make_scatter_key_lines(plot_df):
    key_df = plot_df.sort_values("noisy_rank").copy()
    lines = []
    for _, row in key_df.iterrows():
        point_id = str(row["point_id"])
        species = str(row["kept_species"])
        lines.append(f"{point_id}  {species}")
    return lines


def add_scatter_point_id_labels(ax, plot_df, font_size=None):
    label_font_size = SCATTER_POINT_LABEL_FONT_SIZE if font_size is None else font_size

    for _, row in plot_df.iterrows():
        ax.annotate(
            str(row["point_id"]),
            (row["clean_mse_scaled_mean"], row["noisy_mse_scaled_mean"]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=label_font_size,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.70},
        )


def add_scatter_right_side_key(ax_key, plot_df, architecture_text, noise_percent):
    ax_key.axis("off")

    title = (
        "Point key\n"
        f"Noise: {format_scatter_noise_percent(noise_percent)}\n"
        f"Architecture: {architecture_text}\n"
        "ID  Species combination"
    )
    key_lines = make_scatter_key_lines(plot_df)
    key_text = "\n".join(key_lines)

    ax_key.text(
        0.0,
        1.0,
        title,
        transform=ax_key.transAxes,
        fontsize=SCATTER_KEY_TITLE_FONT_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax_key.text(
        0.0,
        0.86,
        key_text,
        transform=ax_key.transAxes,
        fontsize=SCATTER_KEY_FONT_SIZE,
        family="monospace",
        va="top",
        ha="left",
        linespacing=1.08,
    )


def make_scatter_plot(scatter_df, architecture_text, noise_percent, output_root):
    plot_df, x, y = finite_positive_scatter_xy(scatter_df)

    if plot_df.empty:
        print(
            f"Skipping plot {format_scatter_noise_percent(noise_percent)} / {architecture_text}: "
            "no finite positive points available for plotting."
        )
        return []

    lower, upper = get_scatter_axis_limits(x, y)

    if SCATTER_SHOW_RIGHT_SIDE_KEY:
        fig = plt.figure(figsize=(14.0, 7.2), constrained_layout=True)
        grid = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3.1, 1.65])
        ax = fig.add_subplot(grid[0, 0])
        ax_key = fig.add_subplot(grid[0, 1])
    else:
        fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
        ax_key = None

    ax.scatter(
        plot_df["clean_mse_scaled_mean"],
        plot_df["noisy_mse_scaled_mean"],
        s=55,
        alpha=0.85,
    )

    if SCATTER_SHOW_DIAGONAL_Y_EQUALS_X and lower is not None and upper is not None:
        ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.2, label="y = x")

    if SCATTER_USE_LOG_AXES:
        ax.set_xscale("log")
        ax.set_yscale("log")

    if lower is not None and upper is not None:
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)

    add_scatter_zoom_inset(ax, plot_df)

    if SCATTER_LABEL_POINTS_WITH_IDS:
        add_scatter_point_id_labels(ax, plot_df)

    if SCATTER_SHOW_RIGHT_SIDE_KEY and ax_key is not None:
        add_scatter_right_side_key(ax_key, plot_df, architecture_text, noise_percent)

    ax.set_xlabel("Clean scaled MSE at 0% noise")
    ax.set_ylabel(f"Noisy scaled MSE at {format_scatter_noise_percent(noise_percent)} noise")
    ax.set_title(
        "Clean MSE vs noisy MSE\n"
        f"Noise: {format_scatter_noise_percent(noise_percent)} | Architecture: {architecture_text}"
    )
    ax.grid(True, which="both", alpha=0.25)
    if SCATTER_SHOW_DIAGONAL_Y_EQUALS_X:
        ax.legend(frameon=False)

    base_name = scatter_base_filename(noise_percent, architecture_text)
    saved_paths = []

    if SCATTER_SAVE_PNG:
        png_path = Path(output_root) / f"{base_name}.png"
        fig.savefig(png_path, dpi=SCATTER_DPI, bbox_inches="tight")
        saved_paths.append(str(png_path))

    if SCATTER_SAVE_PDF:
        pdf_path = Path(output_root) / f"{base_name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        saved_paths.append(str(pdf_path))

    plt.close(fig)
    return saved_paths


def save_scatter_plot_data(scatter_df, architecture_text, noise_percent, output_root):
    base_name = scatter_base_filename(noise_percent, architecture_text)
    data_path = Path(output_root) / "data" / f"{base_name}.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    scatter_df.to_csv(data_path, index=False)
    return str(data_path)


def save_noise_scatter_plots(noise_results_root, aggregate_df=None):
    noise_results_root = Path(noise_results_root)

    if aggregate_df is None:
        df, source_label = load_scatter_input_table(noise_results_root)
    else:
        df = aggregate_df.copy()
        source_label = "in-memory aggregate_df from current run"

    df = normalize_scatter_input_table(df, source_label)

    architectures = choose_scatter_architectures(df)
    noise_percents = choose_scatter_noise_percents(df)

    if not architectures:
        raise RuntimeError("No architectures available for scatter plots.")
    if not noise_percents:
        raise RuntimeError("No nonzero noise percentages available for scatter plots.")

    scatter_root = scatter_root_path(noise_results_root)
    scatter_root.mkdir(parents=True, exist_ok=True)

    all_scatter_rows = []
    manifest = {
        "source_csv": source_label,
        "scatter_root": str(scatter_root),
        "metric_mean": SCATTER_METRIC_MEAN,
        "metric_std": SCATTER_METRIC_STD,
        "x_axis": "test_mse_scaled_mean at 0% noise",
        "y_axis": "test_mse_scaled_mean at nonzero noise percentage",
        "point_identification": "numeric point_id labels on the scatter plot plus a right-side species key in the same figure",
        "use_log_axes": SCATTER_USE_LOG_AXES,
        "filename_policy": "<noise_label>__<architecture_token>.png/pdf, e.g. 0.5%__50_50.png",
        "created_plots": [],
        "created_data_files": [],
        "skipped_blocks": [],
    }

    for architecture_text in architectures:
        for noise_percent in noise_percents:
            scatter_df = build_scatter_dataframe(df, architecture_text, noise_percent)

            block_label = f"{format_scatter_noise_percent(noise_percent)} | {architecture_text}"
            if scatter_df.empty:
                manifest["skipped_blocks"].append(block_label)
                continue

            saved_plot_paths = make_scatter_plot(
                scatter_df=scatter_df,
                architecture_text=architecture_text,
                noise_percent=noise_percent,
                output_root=scatter_root,
            )

            if not saved_plot_paths:
                manifest["skipped_blocks"].append(block_label)
                continue

            manifest["created_plots"].extend(saved_plot_paths)
            print(f"Saved scatter plot(s) for {block_label}:")
            for path in saved_plot_paths:
                print(f"  {path}")

            if SCATTER_SAVE_PER_PLOT_DATA_CSV:
                data_path = save_scatter_plot_data(scatter_df, architecture_text, noise_percent, scatter_root)
                manifest["created_data_files"].append(data_path)

            all_scatter_rows.append(scatter_df)

    if all_scatter_rows and SCATTER_SAVE_GLOBAL_DATA_CSV:
        all_scatter_df = pd.concat(all_scatter_rows, ignore_index=True, sort=False)
        global_data_path = scatter_root / "scatterplot_points_all.csv"
        all_scatter_df.to_csv(global_data_path, index=False)
        manifest["global_scatter_data_csv"] = str(global_data_path)
    else:
        manifest["global_scatter_data_csv"] = None

    manifest_path = scatter_root / "scatterplot_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print("\nFinished creating scatter plots.")
    print(f"Scatter plot folder:\n{scatter_root}")
    print(f"Manifest:\n{manifest_path}")
    print(f"Number of plot files created: {len(manifest['created_plots'])}")

    return manifest


# ======================================================================================
# MAIN WORKFLOW
# ======================================================================================


def save_run_info(noise_results_root, total_requested, total_existing, total_to_run):
    run_info = {
        "scheme": SCHEME,
        "base_experiment_name": BASE_EXPERIMENT_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "evaluation_mode": "seed_ensemble",
        "ensemble_definition": "For each species/architecture/noise_repeat, predictions from the 20 neural-network seeds are averaged first; the MSE is then computed from that ensemble-mean prediction.",
        "saved_weights_root": str(SAVED_WEIGHTS_ROOT),
        "noise_results_root": str(noise_results_root),
        "species_configs_requested": SPECIES_CONFIGS,
        "architectures_requested": [list(a) for a in ARCHITECTURES],
        "ensemble_seeds_requested": ENSEMBLE_SEEDS,
        "num_ensemble_seeds_requested": int(len(ENSEMBLE_SEEDS)),
        "noise_stds_requested": NOISE_STDS,
        "noise_labels_requested": [noise_label(s) for s in NOISE_STDS],
        "noise_repeats": NOISE_REPEATS,
        "noise_base_seed": NOISE_BASE_SEED,
        "noise_seed_policy": "noise_rng_seed depends on noise_std and noise_repeat only, not on neural-network seed, so all ensemble members see the same noisy input realization",
        "main_metric": "test_mse_scaled computed on the ensemble-mean scaled K prediction",
        "ranking_metric": RANKING_METRIC_MEAN,
        "ranking_blocks": "noise_percent + architecture",
        "noise_type": "multiplicative Gaussian noise applied equally to all selected unscaled input-density features",
        "resample_negative_densities": RESAMPLE_NEGATIVE_DENSITIES,
        "max_noise_resample_attempts": MAX_NOISE_RESAMPLE_ATTEMPTS,
        "cache_policy": "load compatible saved_weights models for every ensemble seed; otherwise train and save before evaluation",
        "append_policy": (
            "new ensemble rows are appended to each species noise_results.csv; duplicate rows for the same "
            "species/architecture/noise/repeat are de-duplicated keeping the newest row"
        ),
        "skip_existing_evaluations": SKIP_EXISTING_EVALUATIONS,
        "total_requested_ensemble_evaluations": int(total_requested),
        "total_existing_requested_ensemble_evaluations_skipped": int(total_existing),
        "total_new_ensemble_evaluations_to_run": int(total_to_run),
    }
    save_json(Path(noise_results_root) / "noise_ranking_run_info.json", run_info)


def load_or_train_ensemble_models(
    kept_species,
    hidden_size,
    dataset_train,
    dataset_test,
):
    models_by_seed = {}
    training_records_by_seed = {}

    for seed in ENSEMBLE_SEEDS:
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
        models_by_seed[int(seed)] = model
        training_records_by_seed[int(seed)] = training_record

    return models_by_seed, training_records_by_seed


def run_noise_evaluations_and_rankings():
    validate_all_species_configs(SPECIES_CONFIGS)
    validate_architectures(ARCHITECTURES)

    NOISE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    task_plan, total_requested, total_existing, total_to_run = plan_all_tasks(NOISE_RESULTS_ROOT)
    save_run_info(NOISE_RESULTS_ROOT, total_requested, total_existing, total_to_run)

    print("Requested ensemble evaluations:", total_requested)
    print("Already existing requested ensemble evaluations:", total_existing)
    print("New ensemble evaluations to run:", total_to_run)
    print(f"Ensemble seeds: {ENSEMBLE_SEEDS}")
    print(f"Noise results root: {NOISE_RESULTS_ROOT}")

    current_results = []

    if total_to_run > 0:
        with tqdm(total=total_to_run, desc="Ensemble noise ranking evaluations") as pbar:
            for species_name, species_plan in task_plan.items():
                kept_species = species_plan["kept_species"]
                tasks_to_run = species_plan["tasks_to_run"]

                if not tasks_to_run:
                    print(f"Skipping {species_name}: all requested ensemble rows already exist.")
                    continue

                dataset_train, dataset_test = load_datasets_with_saved_scalers(SCHEME, kept_species)

                tasks_df = pd.DataFrame(tasks_to_run)
                for hidden_size_text, group in tasks_df.groupby("hidden_size_text", sort=False):
                    hidden_size = tuple(int(x.strip()) for x in str(hidden_size_text).split(",") if x.strip())

                    models_by_seed, training_records_by_seed = load_or_train_ensemble_models(
                        kept_species=kept_species,
                        hidden_size=hidden_size,
                        dataset_train=dataset_train,
                        dataset_test=dataset_test,
                    )

                    for _, task in group.iterrows():
                        row = run_noise_for_seed_ensemble(
                            models_by_seed=models_by_seed,
                            training_records_by_seed=training_records_by_seed,
                            dataset_test=dataset_test,
                            kept_species=kept_species,
                            hidden_size=hidden_size,
                            noise_std=float(task["noise_std"]),
                            noise_repeat=int(task["noise_repeat"]),
                        )
                        current_results.append(row)

                        pbar.set_postfix(
                            species=species_name,
                            arch=hidden_size_text,
                            noise=row["noise_label"],
                            repeat=int(task["noise_repeat"]),
                            mse=f"{row['test_mse_scaled']:.3e}",
                        )
                        pbar.update(1)
    else:
        print("No new ensemble evaluations required. Rebuilding global files/rankings from existing species folders.")

    append_current_results_to_species_folders(NOISE_RESULTS_ROOT, current_results)

    if REBUILD_GLOBAL_FILES_FROM_ALL_SPECIES_FOLDERS:
        _, aggregate_df = rebuild_fullrun_noise_files_from_direct_species_folders(NOISE_RESULTS_ROOT)
    else:
        aggregate_path = NOISE_RESULTS_ROOT / "fullrun_noise_aggregate_summary.csv"
        if not aggregate_path.exists():
            raise FileNotFoundError(
                f"{aggregate_path} does not exist. Enable REBUILD_GLOBAL_FILES_FROM_ALL_SPECIES_FOLDERS=True "
                "or run once with global rebuild enabled."
            )
        aggregate_df = pd.read_csv(aggregate_path)

    if not aggregate_df.empty:
        ranking_df = save_ranking_outputs(NOISE_RESULTS_ROOT, aggregate_df)
        print(f"Ranking rows created: {len(ranking_df)}")
    else:
        print("No aggregate rows available; rankings were not created.")

    if SAVE_SCATTER_PLOTS_AFTER_RUN:
        if aggregate_df.empty:
            print("Skipping scatter plots: no aggregate rows available.")
        else:
            try:
                save_noise_scatter_plots(NOISE_RESULTS_ROOT, aggregate_df=aggregate_df)
            except Exception as exc:
                print(f"WARNING: Scatter-plot generation failed after ranking completed: {exc}")

    print(f"Ensemble noise results root: {NOISE_RESULTS_ROOT}")


def main():
    run_noise_evaluations_and_rankings()


if __name__ == "__main__":
    main()
