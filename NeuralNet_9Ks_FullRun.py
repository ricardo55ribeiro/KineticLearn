from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ======================================================================================
# Setup
# ======================================================================================

SCHEME = "O2_simple_9K"
EXPERIMENT_NAME = "RunDefault9K"

PRESSURE_CONFIGS = [["1", "2", "5", "10"]]
SPECIES = ["O2(X)", "O2(a)", "O(3P)"]

K_NAMES = [f"K{i}" for i in range(1, 10)]
K_REACTIONS = [
    "e + O2(X) -> e + O2(a)",
    "e + O2(a) -> e + O2(X)",
    "e + O2(X) -> e + 2O(3P)",
    "e + O2(a) -> e + 2O(3P)",
    "O2(a) + O(3P) -> O2(X) + O(3P)",
    "O2(a) + O(3P) + O2(X) -> O2(X) + O(3P) + O2(X)",
    "2O(3P) + O2(X) -> 2O2(X)",
    "O2(a) + wall -> O2(X)",
    "O(3P) + wall -> 0.5O2(X)",
]

ARCHITECTURES = [
    (100, 100, 100),
    (30, 30),
    (50, 50),
    (30, 30, 30),
]

SEEDS = list(range(32, 52))

ACTIVATION = "tanh"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
MAX_EPOCHS = 5000
PATIENCE = 100

TEST_SPLIT = 0.10
VAL_SPLIT = 0.20

LOG10_X = True
LOG10_Y = True
STANDARDIZE_X = True
STANDARDIZE_Y = True

SAVE_WEIGHTS_ROOT = Path("saved_weights")
BASE_RESULTS_DIR = Path("Results_NN")

DATA_SEARCH_ROOT = Path(".")
DATA_FLAT_FILE = Path("9Ks_Dataset") / "O2_simple" / "O2_simple_uniform_seed10_rebuilt_4p.txt"

FORCE_RETRAIN = False
SAVE_SUMMARY_FILES = True
MAX_MODELS_TO_TRAIN: Optional[int] = None
VERBOSE_EPOCH_LOSSES = False
DETERMINISTIC_TORCH = True

SAVE_PLOTS = True
SAVE_PREDICTIONS_CSV = True
SAVE_METRICS_TXT = True
LOSS_SMOOTHING_WINDOW = 25
HASH_TAG_LENGTH = 16


# ======================================================================================
# Constants/utilities
# ======================================================================================

PRESSURE_TO_PA = {
    "1": 133.33,
    "2": 266.66,
    "5": 666.66,
    "10": 1333.30,
}
FLAT_COLUMNS = K_NAMES + ["pressure_Pa"] + SPECIES


class ConfigError(ValueError):
    pass


def normalize_pressure_label(value: object) -> str:
    text = str(value).strip().lower().replace("torr", "").replace(" ", "")
    pa_aliases = {
        "133.33": "1", "133.3300": "1", "1.3333e+02": "1",
        "266.66": "2", "266.6600": "2", "2.6666e+02": "2",
        "666.66": "5", "666.6600": "5", "6.6666e+02": "5",
        "1333.3": "10", "1333.30": "10", "1333.3000": "10", "1.3333e+03": "10",
    }
    if text in pa_aliases:
        return pa_aliases[text]

    try:
        number = float(text)
    except ValueError:
        number = math.nan

    if not math.isnan(number):
        for label in ("1", "2", "5", "10"):
            if abs(number - float(label)) < 1e-12:
                return label
        for label, pa in PRESSURE_TO_PA.items():
            if abs(number - pa) <= 1e-2:
                return label

    raise ConfigError(f"Invalid pressure label {value!r}. Use a subset of {list(PRESSURE_TO_PA)}.")


def normalize_pressures(pressures: Sequence[object]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for pressure in pressures:
        label = normalize_pressure_label(pressure)
        if label not in seen:
            normalized.append(label)
            seen.add(label)
    if not normalized:
        raise ConfigError("At least one pressure must be selected.")
    return normalized


def pressures_folder_name(pressures: Sequence[str]) -> str:
    return "pressures_" + "_".join(normalize_pressures(pressures))


def arch_folder_name(hidden_size: Sequence[int]) -> str:
    return "arch_" + "_".join(str(int(v)) for v in hidden_size)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC_TORCH:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def short_hash_tag(prefix: str, signature: str) -> str:
    return f"{prefix}_{signature[:HASH_TAG_LENGTH]}"


def moving_average(x: Sequence[float], window: int = LOSS_SMOOTHING_WINDOW) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if window <= 1 or len(x_arr) == 0:
        return x_arr
    window = min(int(window), len(x_arr))
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_padded = np.pad(x_arr, (pad_left, pad_right), mode="edge")
    return np.convolve(x_padded, kernel, mode="valid")


def find_flat_dataset_file() -> Path:
    direct = DATA_SEARCH_ROOT / DATA_FLAT_FILE
    if direct.exists():
        return direct.resolve()
    matches = list(DATA_SEARCH_ROOT.glob("**/O2_simple_uniform_seed10_rebuilt_4p.txt"))
    if matches:
        matches.sort(key=lambda p: (len(str(p)), str(p)))
        return matches[0].resolve()
    raise FileNotFoundError(
        "Could not find O2_simple_uniform_seed10_rebuilt_4p.txt.\n"
        f"Looked under: {DATA_SEARCH_ROOT.resolve()}\n"
        f"Expected path: {direct}\n"
        "Place the extracted 9Ks_Dataset folder in the project root, or update DATA_FLAT_FILE."
    )


# ======================================================================================
# Preprocessing
# ======================================================================================

@dataclass
class StandardScalerNP:
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-12) -> "StandardScalerNP":
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < eps, 1.0, scale)
        return cls(mean_=mean.astype(np.float64), scale_=scale.astype(np.float64))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale_ + self.mean_

    def to_dict(self) -> dict:
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScalerNP":
        return cls(
            mean_=np.array(d["mean"], dtype=np.float64),
            scale_=np.array(d["scale"], dtype=np.float64),
        )


@dataclass
class Preprocessor:
    log10_x: bool
    log10_y: bool
    standardize_x: bool
    standardize_y: bool
    x_scaler: Optional[StandardScalerNP]
    y_scaler: Optional[StandardScalerNP]

    @staticmethod
    def _safe_log10(x: np.ndarray, name: str) -> np.ndarray:
        if np.any(x <= 0):
            raise ValueError(f"Cannot log10-transform {name}: found non-positive value {float(np.min(x))}.")
        return np.log10(x)

    @classmethod
    def fit(
        cls,
        x_train_raw: np.ndarray,
        y_train_raw: np.ndarray,
        *,
        log10_x: bool,
        log10_y: bool,
        standardize_x: bool,
        standardize_y: bool,
    ) -> "Preprocessor":
        x_work = cls._safe_log10(x_train_raw, "X") if log10_x else x_train_raw.astype(np.float64)
        y_work = cls._safe_log10(y_train_raw, "y") if log10_y else y_train_raw.astype(np.float64)
        x_scaler = StandardScalerNP.fit(x_work) if standardize_x else None
        y_scaler = StandardScalerNP.fit(y_work) if standardize_y else None
        return cls(log10_x, log10_y, standardize_x, standardize_y, x_scaler, y_scaler)

    def transform_x(self, x_raw: np.ndarray) -> np.ndarray:
        x = self._safe_log10(x_raw, "X") if self.log10_x else x_raw.astype(np.float64)
        if self.x_scaler is not None:
            x = self.x_scaler.transform(x)
        return x.astype(np.float32)

    def transform_y(self, y_raw: np.ndarray) -> np.ndarray:
        y = self._safe_log10(y_raw, "y") if self.log10_y else y_raw.astype(np.float64)
        if self.y_scaler is not None:
            y = self.y_scaler.transform(y)
        return y.astype(np.float32)

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        y = y_scaled.astype(np.float64)
        if self.y_scaler is not None:
            y = self.y_scaler.inverse_transform(y)
        if self.log10_y:
            y = np.power(10.0, y)
        return y

    def to_dict(self) -> dict:
        return {
            "log10_x": self.log10_x,
            "log10_y": self.log10_y,
            "standardize_x": self.standardize_x,
            "standardize_y": self.standardize_y,
            "x_scaler": None if self.x_scaler is None else self.x_scaler.to_dict(),
            "y_scaler": None if self.y_scaler is None else self.y_scaler.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        return cls(
            log10_x=bool(d["log10_x"]),
            log10_y=bool(d["log10_y"]),
            standardize_x=bool(d["standardize_x"]),
            standardize_y=bool(d["standardize_y"]),
            x_scaler=None if d.get("x_scaler") is None else StandardScalerNP.from_dict(d["x_scaler"]),
            y_scaler=None if d.get("y_scaler") is None else StandardScalerNP.from_dict(d["y_scaler"]),
        )


# ======================================================================================
# Dataset loading/grouping
# ======================================================================================

@dataclass
class GroupedDataset:
    pressures: List[str]
    x_raw: np.ndarray
    y_raw: np.ndarray
    sample_ids: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    raw_table_path: Path
    raw_table_sha256: str

    @property
    def n_samples(self) -> int:
        return int(self.x_raw.shape[0])

    @property
    def input_size(self) -> int:
        return int(self.x_raw.shape[1])

    @property
    def output_size(self) -> int:
        return int(self.y_raw.shape[1])


def load_flat_table(path: Path) -> pd.DataFrame:
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] != 13:
        raise ValueError(f"Expected flat dataset with shape (n, 13), got {arr.shape} from {path}")
    return pd.DataFrame(arr, columns=FLAT_COLUMNS)


def pressure_label_from_pa(pa: float, tolerance: float = 1e-2) -> str:
    for label, expected_pa in PRESSURE_TO_PA.items():
        if abs(pa - expected_pa) <= tolerance:
            return label
    raise ValueError(f"Unknown pressure value in dataset: {pa}")


def build_grouped_dataset(flat_path: Path, pressures: Sequence[object]) -> GroupedDataset:
    pressures_norm = normalize_pressures(pressures)
    df = load_flat_table(flat_path).copy()
    df["pressure_label"] = [pressure_label_from_pa(float(v)) for v in df["pressure_Pa"].to_numpy()]

    available_pressures = sorted(df["pressure_label"].unique(), key=lambda p: PRESSURE_TO_PA[p])
    missing = [p for p in pressures_norm if p not in available_pressures]
    if missing:
        raise ConfigError(f"Requested pressures {missing} are not available. Available: {available_pressures}")

    counts = df.groupby("pressure_label").size().to_dict()
    if len(set(counts.values())) != 1:
        raise ValueError(f"Expected same number of rows per pressure. Counts: {counts}")
    n_per_pressure = next(iter(counts.values()))

    by_pressure: Dict[str, pd.DataFrame] = {}
    for p in available_pressures:
        block = df[df["pressure_label"] == p].copy().reset_index(drop=True)
        if len(block) != n_per_pressure:
            raise ValueError(f"Pressure {p} has {len(block)} rows, expected {n_per_pressure}.")
        by_pressure[p] = block

    ref_k = by_pressure[pressures_norm[0]][K_NAMES].to_numpy(dtype=np.float64)
    for p in pressures_norm[1:]:
        this_k = by_pressure[p][K_NAMES].to_numpy(dtype=np.float64)
        if not np.allclose(ref_k, this_k, rtol=0.0, atol=0.0):
            max_diff = float(np.max(np.abs(ref_k - this_k)))
            raise ValueError(f"K values are not exactly aligned across pressure blocks. Pressure {p}, max abs diff = {max_diff}.")

    x_parts = []
    feature_names = []
    for p in pressures_norm:
        x_parts.append(by_pressure[p][SPECIES].to_numpy(dtype=np.float64))
        feature_names.extend([f"{species}@{p}Torr" for species in SPECIES])

    if len(pressures_norm) < 3:
        print("WARNING: fewer than 3 pressures selected. This is likely underdetermined for 9 K targets.")

    return GroupedDataset(
        pressures=pressures_norm,
        x_raw=np.concatenate(x_parts, axis=1),
        y_raw=ref_k.copy(),
        sample_ids=np.arange(n_per_pressure, dtype=np.int64),
        feature_names=feature_names,
        target_names=K_NAMES.copy(),
        raw_table_path=flat_path,
        raw_table_sha256=sha256_file(flat_path),
    )


def split_indices(n_samples: int, seed: int, test_split: float, val_split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_split < 1.0):
        raise ConfigError("TEST_SPLIT must be in (0, 1).")
    if not (0.0 <= val_split < 1.0):
        raise ConfigError("VAL_SPLIT must be in [0, 1).")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(indices)

    n_test = max(1, int(round(test_split * n_samples)))
    test_idx = indices[:n_test]
    train_val_idx = indices[n_test:]

    n_val = int(round(val_split * len(train_val_idx)))
    if val_split > 0:
        n_val = max(1, n_val)
    n_val = min(n_val, max(0, len(train_val_idx) - 1))

    val_idx = train_val_idx[:n_val]
    train_idx = train_val_idx[n_val:]
    if len(train_idx) == 0:
        raise ConfigError("Split produced an empty training set. Reduce TEST_SPLIT/VAL_SPLIT.")
    return train_idx, val_idx, test_idx


# ======================================================================================
# Model
# ======================================================================================

class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: Sequence[int], activation: str):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_size
        for width in hidden_size:
            width = int(width)
            if width <= 0:
                raise ConfigError(f"Invalid hidden layer width: {width}")
            layers.append(nn.Linear(prev, width))
            layers.append(make_activation(activation))
            prev = width
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_activation(name: str) -> nn.Module:
    name = name.strip().lower()
    if name == "tanh": return nn.Tanh()
    if name == "relu": return nn.ReLU()
    if name == "leaky_relu": return nn.LeakyReLU(negative_slope=0.01)
    if name == "sigmoid": return nn.Sigmoid()
    if name == "elu": return nn.ELU()
    if name == "gelu": return nn.GELU()
    raise ConfigError(f"Unknown activation {name!r}.")


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ======================================================================================
# Paths/signatures/cache compatibility
# ======================================================================================

def saved_scheme_root() -> Path:
    return SAVE_WEIGHTS_ROOT / SCHEME


def saved_pressure_root(pressures: Sequence[str]) -> Path:
    return saved_scheme_root() / pressures_folder_name(pressures)


def saved_model_dir(pressures: Sequence[str], hidden_size: Sequence[int], seed: int, config_tag: str) -> Path:
    return saved_pressure_root(pressures) / arch_folder_name(hidden_size) / f"seed_{int(seed):04d}" / config_tag


def results_experiment_root(run_tag: str) -> Path:
    return BASE_RESULTS_DIR / SCHEME / EXPERIMENT_NAME / run_tag


def results_pressure_root(pressures: Sequence[str], run_tag: str) -> Path:
    return results_experiment_root(run_tag) / pressures_folder_name(pressures)


def results_model_dir(pressures: Sequence[str], hidden_size: Sequence[int], seed: int, config_tag: str, run_tag: str) -> Path:
    return results_pressure_root(pressures, run_tag) / f"seed_{int(seed):04d}" / arch_folder_name(hidden_size) / config_tag


def make_model_config_signature(dataset: GroupedDataset, hidden_size: Sequence[int], seed: int) -> Tuple[dict, str, str]:
    config = {
        "scheme": SCHEME,
        "pressures": dataset.pressures,
        "species": SPECIES,
        "k_names": K_NAMES,
        "input_size": dataset.input_size,
        "output_size": dataset.output_size,
        "hidden_size": list(map(int, hidden_size)),
        "seed": int(seed),
        "activation": ACTIVATION,
        "learning_rate": float(LEARNING_RATE),
        "batch_size": int(BATCH_SIZE),
        "max_epochs": int(MAX_EPOCHS),
        "patience": int(PATIENCE),
        "test_split": float(TEST_SPLIT),
        "val_split": float(VAL_SPLIT),
        "log10_x": bool(LOG10_X),
        "log10_y": bool(LOG10_Y),
        "standardize_x": bool(STANDARDIZE_X),
        "standardize_y": bool(STANDARDIZE_Y),
        "raw_table_sha256": dataset.raw_table_sha256,
    }
    sig = stable_json_hash(config)
    return config, sig, short_hash_tag("cfg", sig)


def make_run_config_signature(flat_path: Path, flat_sha256: str, pressure_configs: Sequence[Sequence[str]]) -> Tuple[dict, str, str]:
    config = {
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "flat_dataset_path": str(flat_path),
        "flat_dataset_sha256": flat_sha256,
        "pressure_configs": [normalize_pressures(p) for p in pressure_configs],
        "species": SPECIES,
        "k_names": K_NAMES,
        "architectures": [list(map(int, a)) for a in ARCHITECTURES],
        "seeds": [int(s) for s in SEEDS],
        "activation": ACTIVATION,
        "learning_rate": float(LEARNING_RATE),
        "batch_size": int(BATCH_SIZE),
        "max_epochs": int(MAX_EPOCHS),
        "patience": int(PATIENCE),
        "test_split": float(TEST_SPLIT),
        "val_split": float(VAL_SPLIT),
        "log10_x": bool(LOG10_X),
        "log10_y": bool(LOG10_Y),
        "standardize_x": bool(STANDARDIZE_X),
        "standardize_y": bool(STANDARDIZE_Y),
        "max_models_to_train": MAX_MODELS_TO_TRAIN,
        "save_plots": bool(SAVE_PLOTS),
        "save_predictions_csv": bool(SAVE_PREDICTIONS_CSV),
        "save_metrics_txt": bool(SAVE_METRICS_TXT),
    }
    sig = stable_json_hash(config)
    return config, sig, short_hash_tag("run", sig)


def is_compatible_saved_model(weights_dir: Path, expected_signature: str) -> bool:
    # Crucial rule: cache compatibility depends only on saved_weights, never Results_NN.
    required = [
        weights_dir / "model.pth",
        weights_dir / "model_cache_info.json",
        weights_dir / "scalers.json",
        weights_dir / "split_indices.npz",
        weights_dir / "loss_history.csv",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        info = read_json(weights_dir / "model_cache_info.json")
    except Exception:
        return False
    return info.get("config_signature") == expected_signature


# ======================================================================================
# Metrics/plot/report helpers
# ======================================================================================

def compute_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray, y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray) -> dict:
    err_scaled = y_pred_scaled - y_true_scaled
    mse_scaled = float(np.mean(err_scaled ** 2))
    y_true_log10 = np.log10(y_true_raw)
    y_pred_log10 = np.log10(y_pred_raw)
    err_log10 = y_pred_log10 - y_true_log10
    rel = np.abs((y_pred_raw - y_true_raw) / y_true_raw)
    mean_rel_per_k = rel.mean(axis=0) * 100.0
    max_rel_per_k = rel.max(axis=0) * 100.0
    return {
        "test_mse_scaled": mse_scaled,
        "test_rmse_scaled": float(np.sqrt(mse_scaled)),
        "test_mse_log10": float(np.mean(err_log10 ** 2)),
        "test_rmse_log10": float(np.sqrt(np.mean(err_log10 ** 2))),
        "mean_relative_error_percent": float(mean_rel_per_k.mean()),
        "max_relative_error_percent": float(max_rel_per_k.max()),
        "mean_relative_error_percent_per_k": {name: float(v) for name, v in zip(K_NAMES, mean_rel_per_k)},
        "max_relative_error_percent_per_k": {name: float(v) for name, v in zip(K_NAMES, max_rel_per_k)},
    }


def _set_matplotlib_backend() -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)


def history_records_from_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if not {"epoch", "train_loss", "val_loss"}.issubset(df.columns):
        return []
    return df[["epoch", "train_loss", "val_loss"]].to_dict(orient="records")


def save_loss_history_csv(output_dir: Path, loss_history: List[dict]) -> None:
    if not loss_history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(loss_history)
    df["train_loss_smooth"] = moving_average(df["train_loss"].to_numpy(), window=LOSS_SMOOTHING_WINDOW)
    df["val_loss_smooth"] = moving_average(df["val_loss"].to_numpy(), window=LOSS_SMOOTHING_WINDOW)
    df.to_csv(output_dir / "loss_history.csv", index=False)


def save_predictions_csv(output_dir: Path, y_true_raw: np.ndarray, y_pred_raw: np.ndarray, y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, test_idx: np.ndarray) -> None:
    if not SAVE_PREDICTIONS_CSV:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    data: Dict[str, np.ndarray] = {"sample_id": np.asarray(test_idx, dtype=np.int64)}
    for i, name in enumerate(K_NAMES):
        true_raw = y_true_raw[:, i]
        pred_raw = y_pred_raw[:, i]
        denom = np.where(np.abs(true_raw) < 1e-300, 1e-300, true_raw)
        rel_err = np.abs((pred_raw - true_raw) / denom)
        data[f"{name}_true_scaled"] = y_true_scaled[:, i]
        data[f"{name}_pred_scaled"] = y_pred_scaled[:, i]
        data[f"{name}_true_raw"] = true_raw
        data[f"{name}_pred_raw"] = pred_raw
        data[f"{name}_abs_err"] = np.abs(pred_raw - true_raw)
        data[f"{name}_sq_err"] = (pred_raw - true_raw) ** 2
        data[f"{name}_rel_err"] = rel_err
        data[f"{name}_rel_err_percent"] = rel_err * 100.0
    pd.DataFrame(data).to_csv(output_dir / "predictions.csv", index=False)


def save_metrics_txt(output_dir: Path, info: dict, metrics: dict) -> None:
    if not SAVE_METRICS_TXT:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    model = info.get("model", {})
    split = info.get("split", {})
    training = info.get("training", {})
    dataset = info.get("dataset", {})
    lines = [
        f"Scheme: {info.get('scheme')}",
        f"Status: {info.get('status')}",
        f"Config tag: {info.get('config_tag')}",
        f"Config signature: {info.get('config_signature')}",
        f"Pressures: {dataset.get('pressures')}",
        f"Species: {dataset.get('species')}",
        f"Hidden size: {model.get('hidden_size')}",
        f"Activation: {model.get('activation')}",
        f"Input size: {model.get('input_size')}",
        f"Output size: {model.get('output_size')}",
        f"Number of parameters: {model.get('num_parameters')}",
        f"Seed: {training.get('seed')}",
        f"Train/Val/Test: {split.get('n_train')}/{split.get('n_val')}/{split.get('n_test')}",
        f"Learning rate: {training.get('learning_rate')}",
        f"Batch size: {training.get('batch_size')}",
        f"Max epochs: {training.get('max_epochs')}",
        f"Patience: {training.get('patience')}",
        f"Epochs ran: {training.get('epochs_ran')}",
        f"Best epoch: {training.get('best_epoch')}",
        f"Best validation loss: {training.get('best_val_loss')}",
        f"Final train loss: {training.get('final_train_loss')}",
        f"Final validation loss: {training.get('final_val_loss')}",
        f"Current run training time (s): {training.get('current_run_training_time_s')}",
        f"Cached training time (s): {training.get('cached_training_time_s')}",
        "",
        f"Test MSE scaled: {metrics.get('test_mse_scaled')}",
        f"Test RMSE scaled: {metrics.get('test_rmse_scaled')}",
        f"Test MSE log10: {metrics.get('test_mse_log10')}",
        f"Test RMSE log10: {metrics.get('test_rmse_log10')}",
        f"Mean relative error (%): {metrics.get('mean_relative_error_percent')}",
        f"Max relative error (%): {metrics.get('max_relative_error_percent')}",
        "",
        "Per-K relative errors:",
    ]
    mean_rel = metrics.get("mean_relative_error_percent_per_k", {})
    max_rel = metrics.get("max_relative_error_percent_per_k", {})
    for i, name in enumerate(K_NAMES):
        lines.append(f"  {name}: mean={mean_rel.get(name)} %, max={max_rel.get(name)} % | {K_REACTIONS[i]}")
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_loss_curves(loss_history: List[dict], output_dir: Path, log_scale: bool = True) -> None:
    if not SAVE_PLOTS or not loss_history:
        return
    _set_matplotlib_backend()
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(loss_history)
    epochs = df["epoch"].to_numpy(dtype=float)
    train_smooth = moving_average(df["train_loss"].to_numpy(dtype=float), LOSS_SMOOTHING_WINDOW)
    val_smooth = moving_average(df["val_loss"].to_numpy(dtype=float), LOSS_SMOOTHING_WINDOW)
    plt.rcParams.update({"font.size": 14, "text.usetex": False})
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_smooth, linewidth=1.8, label="Training Loss")
    plt.plot(epochs, val_smooth, linewidth=1.8, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    if log_scale:
        plt.yscale("log")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "NeuralNet_loss_curves.pdf")
    plt.savefig(output_dir / "NeuralNet_loss_curves.png", dpi=200)
    plt.close()


def plot_predictions_grid(output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, *, filename_base: str, title_prefix: str, true_denominator: bool = True) -> None:
    if not SAVE_PLOTS:
        return
    _set_matplotlib_backend()
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    output_size = y_true.shape[1]
    ncols = min(3, output_size)
    nrows = int(math.ceil(output_size / ncols))
    plt.rcParams.update({"font.size": 12, "text.usetex": False})
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(output_size):
        ax = axes[i]
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        ax.scatter(true_i, pred_i, alpha=0.8, s=25)
        finite = np.concatenate([true_i[np.isfinite(true_i)], pred_i[np.isfinite(pred_i)]])
        if finite.size:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            pad = max(abs(vmin) * 0.05, 1e-12) if abs(vmax - vmin) < 1e-15 else 0.05 * (vmax - vmin)
            vmin -= pad
            vmax += pad
            ax.plot([vmin, vmax], [vmin, vmax], "--", color="black", linewidth=1)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
        denom_src = true_i if true_denominator else pred_i
        denom = np.where(np.abs(denom_src) < 1e-300, 1e-300, denom_src)
        rel = np.abs((pred_i - true_i) / denom)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{title_prefix} {K_NAMES[i]}")
        ax.text(
            0.05, 0.95,
            f"Mean δrel={float(np.mean(rel) * 100.0):.2f}%\nMax δrel={float(np.max(rel) * 100.0):.2f}%",
            fontsize=10, transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.25),
        )
        max_index = int(np.argmax(rel))
        ax.scatter(true_i[max_index], pred_i[max_index], color="gold", edgecolor="black", zorder=3, s=45)
    for j in range(output_size, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename_base}.pdf")
    fig.savefig(output_dir / f"{filename_base}.png", dpi=200)
    plt.close(fig)


def plot_results(output_dir: Path, y_true_raw: np.ndarray, y_pred_raw: np.ndarray, y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray) -> None:
    if not SAVE_PLOTS:
        return
    plot_predictions_grid(output_dir, y_true_scaled, y_pred_scaled, filename_base="NeuralNet", title_prefix="Scaled", true_denominator=True)
    if np.all(y_true_raw > 0) and np.all(y_pred_raw > 0):
        plot_predictions_grid(output_dir, np.log10(y_true_raw), np.log10(y_pred_raw), filename_base="NeuralNet_log10_raw", title_prefix="log10 raw", true_denominator=False)


def save_result_artifacts(result_dir: Path, info: dict, metrics: dict, loss_history: List[dict], y_true_raw: np.ndarray, y_pred_raw: np.ndarray, y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, test_idx: np.ndarray) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_dir / "model_info.json", info)
    save_json(result_dir / "metrics.json", metrics)
    save_loss_history_csv(result_dir, loss_history)
    np.savez_compressed(
        result_dir / "test_predictions.npz",
        y_test_true_raw=y_true_raw,
        y_test_pred_raw=y_pred_raw,
        y_test_true_scaled=y_true_scaled,
        y_test_pred_scaled=y_pred_scaled,
        test_idx=test_idx,
    )
    save_predictions_csv(result_dir, y_true_raw, y_pred_raw, y_true_scaled, y_pred_scaled, test_idx)
    save_metrics_txt(result_dir, info, metrics)
    plot_loss_curves(loss_history, result_dir, log_scale=True)
    plot_results(result_dir, y_true_raw, y_pred_raw, y_true_scaled, y_pred_scaled)


# ======================================================================================
# Training/evaluation
# ======================================================================================

def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator, drop_last=False)


def evaluate_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    if len(x) == 0:
        return float("nan")
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32, device=device))
        loss = nn.MSELoss()(pred, torch.tensor(y, dtype=torch.float32, device=device))
    return float(loss.item())


def predict_scaled(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            batch = torch.tensor(x[start:start + 1024], dtype=torch.float32, device=device)
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def load_cached_model(weights_dir: Path, dataset: GroupedDataset, hidden_size: Sequence[int], device: torch.device) -> nn.Module:
    model = MLP(dataset.input_size, dataset.output_size, hidden_size, ACTIVATION).to(device)
    state_dict = torch.load(weights_dir / "model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_info(status: str, dataset: GroupedDataset, hidden_size: Sequence[int], seed: int, config: dict, config_signature: str, config_tag: str, weights_dir: Path, result_dir: Path, model: nn.Module, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, training_info: dict, metrics: dict) -> dict:
    return {
        "script": Path(__file__).name,
        "scheme": SCHEME,
        "status": status,
        "reused_saved_weights": status == "reused",
        "config": config,
        "config_signature": config_signature,
        "config_tag": config_tag,
        "model": {
            "input_size": dataset.input_size,
            "output_size": dataset.output_size,
            "hidden_size": list(map(int, hidden_size)),
            "activation": ACTIVATION,
            "num_parameters": model_parameter_count(model),
        },
        "dataset": {
            "raw_table_path": str(dataset.raw_table_path),
            "raw_table_sha256": dataset.raw_table_sha256,
            "n_grouped_samples": dataset.n_samples,
            "pressures": dataset.pressures,
            "pressure_values_pa": [PRESSURE_TO_PA[p] for p in dataset.pressures],
            "species": SPECIES,
            "feature_names": dataset.feature_names,
            "target_names": dataset.target_names,
            "k_reactions": dict(zip(K_NAMES, K_REACTIONS)),
            "x_raw_shape": list(dataset.x_raw.shape),
            "y_raw_shape": list(dataset.y_raw.shape),
        },
        "split": {
            "seed": int(seed),
            "test_split": float(TEST_SPLIT),
            "val_split": float(VAL_SPLIT),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "effective_train_fraction": float(len(train_idx) / dataset.n_samples),
            "effective_val_fraction": float(len(val_idx) / dataset.n_samples),
            "effective_test_fraction": float(len(test_idx) / dataset.n_samples),
        },
        "training": {
            "seed": int(seed),
            "learning_rate": float(LEARNING_RATE),
            "batch_size": int(BATCH_SIZE),
            "max_epochs": int(MAX_EPOCHS),
            "patience": int(PATIENCE),
            **training_info,
        },
        "metrics": metrics,
        "files": {
            "saved_weights_dir": str(weights_dir),
            "model_pth": str(weights_dir / "model.pth"),
            "model_cache_info_json": str(weights_dir / "model_cache_info.json"),
            "cache_scalers_json": str(weights_dir / "scalers.json"),
            "cache_loss_history_csv": str(weights_dir / "loss_history.csv"),
            "cache_split_indices_npz": str(weights_dir / "split_indices.npz"),
            "results_dir": str(result_dir),
            "model_info_json": str(result_dir / "model_info.json"),
            "metrics_json": str(result_dir / "metrics.json"),
            "metrics_txt": str(result_dir / "metrics.txt"),
            "loss_history_csv": str(result_dir / "loss_history.csv"),
            "predictions_npz": str(result_dir / "test_predictions.npz"),
            "predictions_csv": str(result_dir / "predictions.csv"),
        },
    }


def save_cache_artifacts(weights_dir: Path, cache_info: dict, preprocessor: Preprocessor, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, loss_history: List[dict]) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_json(weights_dir / "model_cache_info.json", cache_info)
    save_json(weights_dir / "scalers.json", preprocessor.to_dict())
    np.savez_compressed(weights_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    save_loss_history_csv(weights_dir, loss_history)


def evaluate_and_write_results_from_cache(dataset: GroupedDataset, hidden_size: Sequence[int], seed: int, config: dict, config_signature: str, config_tag: str, weights_dir: Path, result_dir: Path, device: torch.device) -> dict:
    cache_info = read_json(weights_dir / "model_cache_info.json")
    preprocessor = Preprocessor.from_dict(read_json(weights_dir / "scalers.json"))
    split = np.load(weights_dir / "split_indices.npz")
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
    loss_history = history_records_from_csv(weights_dir / "loss_history.csv")

    x_test_raw = dataset.x_raw[test_idx]
    y_test_raw = dataset.y_raw[test_idx]
    x_test = preprocessor.transform_x(x_test_raw)
    y_test = preprocessor.transform_y(y_test_raw)

    model = load_cached_model(weights_dir, dataset, hidden_size, device)
    y_pred_scaled = predict_scaled(model, x_test, device)
    y_pred_raw = preprocessor.inverse_transform_y(y_pred_scaled)
    metrics = compute_metrics(y_test_raw, y_pred_raw, y_test, y_pred_scaled)

    cached_training = cache_info.get("training", {})
    training_info = {
        "epochs_ran": cached_training.get("epochs_ran"),
        "best_epoch": cached_training.get("best_epoch"),
        "best_train_loss": cached_training.get("best_train_loss"),
        "best_val_loss": cached_training.get("best_val_loss"),
        "final_train_loss": cached_training.get("final_train_loss"),
        "final_val_loss": cached_training.get("final_val_loss"),
        "cached_training_time_s": cached_training.get("cached_training_time_s", cached_training.get("current_run_training_time_s")),
        "current_run_training_time_s": 0.0,
        "device": str(device),
    }
    info = build_info("reused", dataset, hidden_size, seed, config, config_signature, config_tag, weights_dir, result_dir, model, train_idx, val_idx, test_idx, training_info, metrics)
    save_result_artifacts(result_dir, info, metrics, loss_history, y_test_raw, y_pred_raw, y_test, y_pred_scaled, test_idx)
    return {
        "status": "reused",
        "model_dir": str(result_dir),
        "results_dir": str(result_dir),
        "saved_weights_dir": str(weights_dir),
        "saved_weights_path": str(weights_dir / "model.pth"),
        "config_tag": config_tag,
        "config_signature": config_signature,
        "info": info,
        "metrics": metrics,
    }


def train_one_model(dataset: GroupedDataset, hidden_size: Sequence[int], seed: int, device: torch.device, run_tag: str) -> dict:
    config, config_signature, config_tag = make_model_config_signature(dataset, hidden_size, seed)
    weights_dir = saved_model_dir(dataset.pressures, hidden_size, seed, config_tag)
    result_dir = results_model_dir(dataset.pressures, hidden_size, seed, config_tag, run_tag)

    if FORCE_RETRAIN:
        # Only the exact cfg folder is removed; other cfg_* folders are preserved.
        if weights_dir.exists():
            shutil.rmtree(weights_dir)
        if result_dir.exists():
            shutil.rmtree(result_dir)

    weights_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if not FORCE_RETRAIN and is_compatible_saved_model(weights_dir, config_signature):
        return evaluate_and_write_results_from_cache(dataset, hidden_size, seed, config, config_signature, config_tag, weights_dir, result_dir, device)

    set_global_seed(seed)
    train_idx, val_idx, test_idx = split_indices(dataset.n_samples, seed, TEST_SPLIT, VAL_SPLIT)

    x_train_raw, y_train_raw = dataset.x_raw[train_idx], dataset.y_raw[train_idx]
    x_val_raw, y_val_raw = dataset.x_raw[val_idx], dataset.y_raw[val_idx]
    x_test_raw, y_test_raw = dataset.x_raw[test_idx], dataset.y_raw[test_idx]

    preprocessor = Preprocessor.fit(
        x_train_raw, y_train_raw,
        log10_x=LOG10_X, log10_y=LOG10_Y,
        standardize_x=STANDARDIZE_X, standardize_y=STANDARDIZE_Y,
    )
    x_train, y_train = preprocessor.transform_x(x_train_raw), preprocessor.transform_y(y_train_raw)
    x_val, y_val = preprocessor.transform_x(x_val_raw), preprocessor.transform_y(y_val_raw)
    x_test, y_test = preprocessor.transform_x(x_test_raw), preprocessor.transform_y(y_test_raw)

    model = MLP(dataset.input_size, dataset.output_size, hidden_size, ACTIVATION).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(x_train, y_train, BATCH_SIZE, shuffle=True, seed=seed)

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    loss_history: List[dict] = []
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_loss = evaluate_loss(model, x_val, y_val, device) if len(val_idx) else train_loss
        loss_history.append({"epoch": int(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if VERBOSE_EPOCH_LOSSES and (epoch == 1 or epoch % 100 == 0):
            print(f"epoch={epoch:5d} train_loss={train_loss:.6e} val_loss={val_loss:.6e}")

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break

    training_time_s = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)

    final_train_loss = evaluate_loss(model, x_train, y_train, device)
    final_val_loss = evaluate_loss(model, x_val, y_val, device) if len(val_idx) else float("nan")
    y_pred_scaled = predict_scaled(model, x_test, device)
    y_pred_raw = preprocessor.inverse_transform_y(y_pred_scaled)
    metrics = compute_metrics(y_test_raw, y_pred_raw, y_test, y_pred_scaled)

    training_info = {
        "epochs_ran": int(len(loss_history)),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "cached_training_time_s": float(training_time_s),
        "current_run_training_time_s": float(training_time_s),
        "device": str(device),
    }
    info = build_info("trained", dataset, hidden_size, seed, config, config_signature, config_tag, weights_dir, result_dir, model, train_idx, val_idx, test_idx, training_info, metrics)

    # saved_weights: model cache only. No plots/results.
    torch.save(model.state_dict(), weights_dir / "model.pth")
    save_cache_artifacts(weights_dir, info, preprocessor, train_idx, val_idx, test_idx, loss_history)

    # Results_NN: disposable/regenerable outputs.
    save_result_artifacts(result_dir, info, metrics, loss_history, y_test_raw, y_pred_raw, y_test, y_pred_scaled, test_idx)

    return {
        "status": "trained",
        "model_dir": str(result_dir),
        "results_dir": str(result_dir),
        "saved_weights_dir": str(weights_dir),
        "saved_weights_path": str(weights_dir / "model.pth"),
        "config_tag": config_tag,
        "config_signature": config_signature,
        "info": info,
        "metrics": metrics,
    }


# ======================================================================================
# Summaries
# ======================================================================================

def make_summary_row(result: dict, dataset: GroupedDataset, hidden_size: Sequence[int], seed: int) -> dict:
    info = result.get("info", {})
    metrics = result.get("metrics", {}) or info.get("metrics", {}) or {}
    training = info.get("training", {})
    split = info.get("split", {})
    model = info.get("model", {})
    row = {
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "pressures": ",".join(dataset.pressures),
        "pressure_folder": pressures_folder_name(dataset.pressures),
        "num_pressures": len(dataset.pressures),
        "species": ",".join(SPECIES),
        "num_species": len(SPECIES),
        "num_grouped_samples": dataset.n_samples,
        "input_size": dataset.input_size,
        "output_size": dataset.output_size,
        "hidden_size": ",".join(map(str, hidden_size)),
        "arch_folder": arch_folder_name(hidden_size),
        "seed": int(seed),
        "config_tag": result.get("config_tag"),
        "config_signature": result.get("config_signature"),
        "status": result.get("status"),
        "results_dir": result.get("results_dir", result.get("model_dir")),
        "saved_weights_dir": result.get("saved_weights_dir"),
        "saved_weights_path": result.get("saved_weights_path"),
        "num_parameters": model.get("num_parameters"),
        "n_train": split.get("n_train"),
        "n_val": split.get("n_val"),
        "n_test": split.get("n_test"),
        "effective_train_fraction": split.get("effective_train_fraction"),
        "effective_val_fraction": split.get("effective_val_fraction"),
        "effective_test_fraction": split.get("effective_test_fraction"),
        "epochs_ran": training.get("epochs_ran"),
        "best_epoch": training.get("best_epoch"),
        "best_val_loss": training.get("best_val_loss"),
        "final_train_loss": training.get("final_train_loss"),
        "final_val_loss": training.get("final_val_loss"),
        "current_run_training_time_s": training.get("current_run_training_time_s", 0.0),
        "cached_training_time_s": training.get("cached_training_time_s"),
        "test_mse_scaled": metrics.get("test_mse_scaled"),
        "test_rmse_scaled": metrics.get("test_rmse_scaled"),
        "test_mse_log10": metrics.get("test_mse_log10"),
        "test_rmse_log10": metrics.get("test_rmse_log10"),
        "mean_relative_error_percent": metrics.get("mean_relative_error_percent"),
        "max_relative_error_percent": metrics.get("max_relative_error_percent"),
    }
    mean_rel = metrics.get("mean_relative_error_percent_per_k", {})
    max_rel = metrics.get("max_relative_error_percent_per_k", {})
    for name in K_NAMES:
        row[f"{name}_mean_relative_error_percent"] = mean_rel.get(name)
        row[f"{name}_max_relative_error_percent"] = max_rel.get(name)
    return row


def save_global_summary_text(root: Path, rows: List[dict], filename: str = "summary.txt") -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        "Architecture comparison summary", "",
        f"Scheme: {SCHEME}",
        f"Experiment name: {EXPERIMENT_NAME}",
        f"Pressure configs: {[normalize_pressures(p) for p in PRESSURE_CONFIGS]}",
        f"Architectures: {[list(a) for a in ARCHITECTURES]}",
        f"Seeds: {SEEDS}",
        f"Activation: {ACTIVATION}",
        f"Learning rate: {LEARNING_RATE}",
        f"Batch size: {BATCH_SIZE}",
        f"Max epochs: {MAX_EPOCHS}",
        f"Patience: {PATIENCE}",
        f"TEST_SPLIT: {TEST_SPLIT}",
        f"VAL_SPLIT: {VAL_SPLIT} (fraction of remaining train+validation pool)", "",
    ]
    for row in rows:
        lines += [
            f"Seed: {row.get('seed')}",
            f"Architecture: {row.get('hidden_size')}",
            f"  Config tag: {row.get('config_tag')}",
            f"  Pressures: {row.get('pressures')}",
            f"  Status: {row.get('status')}",
            f"  Results dir: {row.get('results_dir')}",
            f"  Saved weights dir: {row.get('saved_weights_dir')}",
            f"  Saved weights: {row.get('saved_weights_path')}",
            f"  Train/Val/Test: {row.get('n_train')}/{row.get('n_val')}/{row.get('n_test')}",
            f"  Epochs ran: {row.get('epochs_ran')}",
            f"  Best epoch: {row.get('best_epoch')}",
            f"  Best val loss: {row.get('best_val_loss')}",
            f"  Current run training time (s): {row.get('current_run_training_time_s')}",
            f"  Cached training time (s): {row.get('cached_training_time_s')}",
            f"  Test MSE scaled: {row.get('test_mse_scaled')}",
            f"  Test RMSE log10: {row.get('test_rmse_log10')}",
            f"  Mean relative error (%): {row.get('mean_relative_error_percent')}",
            f"  Max relative error (%): {row.get('max_relative_error_percent')}", "",
        ]
    with (root / filename).open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_summary_tables(rows: List[dict], run_config: dict, run_signature: str, run_tag: str) -> None:
    root = results_experiment_root(run_tag)
    root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    summary_path = root / "pretrain_summary.csv"
    aggregate_path = root / "pretrain_aggregate_summary.csv"
    info_path = root / "pretrain_info.json"
    text_summary_path = root / "summary.txt"
    df.to_csv(summary_path, index=False)
    if not df.empty:
        aggregate = (
            df.groupby(["scheme", "pressures", "num_pressures", "arch_folder", "config_tag"], as_index=False)
            .agg(
                num_seeds=("seed", "nunique"),
                num_models=("seed", "count"),
                num_trained=("status", lambda s: int((s == "trained").sum())),
                num_reused=("status", lambda s: int((s == "reused").sum())),
                mean_test_mse_scaled=("test_mse_scaled", "mean"),
                std_test_mse_scaled=("test_mse_scaled", "std"),
                mean_test_rmse_scaled=("test_rmse_scaled", "mean"),
                mean_test_rmse_log10=("test_rmse_log10", "mean"),
                mean_relative_error_percent=("mean_relative_error_percent", "mean"),
                std_relative_error_percent=("mean_relative_error_percent", "std"),
                max_relative_error_percent=("max_relative_error_percent", "max"),
                mean_best_val_loss=("best_val_loss", "mean"),
                mean_epochs_ran=("epochs_ran", "mean"),
                total_current_run_training_time_s=("current_run_training_time_s", "sum"),
                mean_cached_training_time_s=("cached_training_time_s", "mean"),
            )
        )
        aggregate.to_csv(aggregate_path, index=False)
    save_global_summary_text(root, rows)
    run_info = {
        "script": Path(__file__).name,
        "purpose": "Train/reuse NN saved weights for the 9-K inverse-problem O2_simple dataset and save diagnostics under Results_NN.",
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "run_tag": run_tag,
        "run_signature": run_signature,
        "run_config": run_config,
        "saved_weights_root": str(SAVE_WEIGHTS_ROOT),
        "scheme_saved_weights_root": str(saved_scheme_root()),
        "results_root": str(BASE_RESULTS_DIR),
        "results_run_root": str(root),
        "force_retrain": bool(FORCE_RETRAIN),
        "cache_policy": "Cache compatibility checks only saved_weights/<SCHEME>/.../<cfg_tag>/, never Results_NN.",
        "saved_weights_cache_files_per_model": ["model.pth", "model_cache_info.json", "scalers.json", "split_indices.npz", "loss_history.csv"],
        "summary_csv": str(summary_path),
        "aggregate_csv": str(aggregate_path),
        "summary_txt": str(text_summary_path),
    }
    save_json(info_path, run_info)


# ======================================================================================
# Main
# ======================================================================================

def main() -> None:
    flat_path = find_flat_dataset_file()
    flat_sha256 = sha256_file(flat_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pressure_configs = [normalize_pressures(p) for p in PRESSURE_CONFIGS]
    run_config, run_signature, run_tag = make_run_config_signature(flat_path, flat_sha256, pressure_configs)

    SAVE_WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_scheme_root().mkdir(parents=True, exist_ok=True)
    results_experiment_root(run_tag).mkdir(parents=True, exist_ok=True)

    planned_models = len(pressure_configs) * len(ARCHITECTURES) * len(SEEDS)
    planned_display = min(planned_models, MAX_MODELS_TO_TRAIN) if MAX_MODELS_TO_TRAIN is not None else planned_models

    print(f"Dataset file: {flat_path}")
    print(f"Dataset SHA256: {flat_sha256}")
    print(f"Saved-weights root: {saved_scheme_root().resolve()}")
    print(f"Results run root: {results_experiment_root(run_tag).resolve()}")
    print(f"Run tag: {run_tag}")
    print(f"Device: {device}")
    print(f"Pressure configs: {pressure_configs}")
    print(f"Architectures: {ARCHITECTURES}")
    print(f"Seeds: {SEEDS[0]} to {SEEDS[-1]} ({len(SEEDS)} seeds)")
    print(f"FORCE_RETRAIN: {FORCE_RETRAIN}")
    print(f"Planned models this run: {planned_display} / {planned_models}\n")

    rows: List[dict] = []
    models_done = 0
    with tqdm(total=planned_display, desc="Training/reusing saved_weights") as pbar:
        for pressures in pressure_configs:
            dataset = build_grouped_dataset(flat_path, pressures)
            #print("")
            print(f"Pressure config: {pressures}")
            print(f"Grouped inverse samples: {dataset.n_samples}")
            print(f"X shape: {dataset.x_raw.shape}")
            print(f"y shape: {dataset.y_raw.shape}")
            print(f"Feature names: {dataset.feature_names}")
            for hidden_size in ARCHITECTURES:
                for seed in SEEDS:
                    if MAX_MODELS_TO_TRAIN is not None and models_done >= MAX_MODELS_TO_TRAIN:
                        break
                    result = train_one_model(dataset, hidden_size, seed, device, run_tag)
                    row = make_summary_row(result, dataset, hidden_size, seed)
                    rows.append(row)
                    rel = row.get("mean_relative_error_percent")
                    rel_text = "nan" if rel is None or pd.isna(rel) else f"{rel:.3g}%"
                    epochs = row.get("epochs_ran")
                    epochs_text = "?" if epochs is None or pd.isna(epochs) else str(int(epochs))
                    pbar.set_postfix(
                        pressures="_".join(pressures),
                        arch="_".join(map(str, hidden_size)),
                        seed=seed,
                        status=result.get("status", "?"),
                        epochs=epochs_text,
                        rel=rel_text,
                        cfg=result.get("config_tag", ""),
                    )
                    pbar.update(1)
                    models_done += 1
                if MAX_MODELS_TO_TRAIN is not None and models_done >= MAX_MODELS_TO_TRAIN:
                    break
            if MAX_MODELS_TO_TRAIN is not None and models_done >= MAX_MODELS_TO_TRAIN:
                break

    if SAVE_SUMMARY_FILES:
        save_summary_tables(rows, run_config, run_signature, run_tag)

    num_trained = sum(1 for r in rows if r.get("status") == "trained")
    num_reused = sum(1 for r in rows if r.get("status") == "reused")
    print("")
    print("Finished saved_weights training/reuse.")
    print(f"Models checked: {len(rows)}")
    print(f"Newly trained models: {num_trained}")
    print(f"Reused compatible models: {num_reused}")
    print(f"Saved weights folder: {saved_scheme_root().resolve()}")
    print(f"Results run folder: {results_experiment_root(run_tag).resolve()}")
    if SAVE_SUMMARY_FILES:
        print(f"Summary CSV: {(results_experiment_root(run_tag) / 'pretrain_summary.csv').resolve()}")
        print(f"Aggregate CSV: {(results_experiment_root(run_tag) / 'pretrain_aggregate_summary.csv').resolve()}")
        print(f"Summary TXT: {(results_experiment_root(run_tag) / 'summary.txt').resolve()}")


if __name__ == "__main__":
    main()
