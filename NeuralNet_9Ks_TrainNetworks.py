from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    (30, 30),
    (50, 50),
    (30, 30, 30),
]

# 20 seeds: 32, 33, ..., 51.
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
FORCE_RETRAIN = False
SAVE_SUMMARY_FILES = True

DATA_SEARCH_ROOT = Path(".")
DATA_FLAT_FILE = Path("9Ks_Dataset") / "O2_simple" / "O2_simple_uniform_seed10_rebuilt_4p.txt"

MAX_MODELS_TO_TRAIN: Optional[int] = None

VERBOSE_EPOCH_LOSSES = False

DETERMINISTIC_TORCH = True


# ======================================================================================
# Constants and utility functions
# ======================================================================================

PRESSURE_TO_PA = {
    "1": 133.33,
    "2": 266.66,
    "5": 666.66,
    "10": 1333.30,
}

PA_TO_PRESSURE = {v: k for k, v in PRESSURE_TO_PA.items()}

FLAT_COLUMNS = K_NAMES + ["pressure_Pa"] + SPECIES


class ConfigError(ValueError):
    """Raised when the run configuration is inconsistent."""


def normalize_pressure_label(value: object) -> str:
    """Normalize pressure labels such as '1', '1 Torr', '1Torr', 1, 1.0."""
    text = str(value).strip().lower()
    text = text.replace("torr", "").replace(" ", "")

    # Accept Pa-like labels too.
    pa_aliases = {
        "133.33": "1",
        "133.3300": "1",
        "1.3333e+02": "1",
        "266.66": "2",
        "266.6600": "2",
        "2.6666e+02": "2",
        "666.66": "5",
        "666.6600": "5",
        "6.6666e+02": "5",
        "1333.3": "10",
        "1333.30": "10",
        "1333.3000": "10",
        "1.3333e+03": "10",
    }
    if text in pa_aliases:
        return pa_aliases[text]

    try:
        number = float(text)
    except ValueError:
        number = math.nan

    if not math.isnan(number):
        if abs(number - 1.0) < 1e-12:
            return "1"
        if abs(number - 2.0) < 1e-12:
            return "2"
        if abs(number - 5.0) < 1e-12:
            return "5"
        if abs(number - 10.0) < 1e-12:
            return "10"

        # Also accept approximate pressure in Pa.
        for label, pa in PRESSURE_TO_PA.items():
            if abs(number - pa) <= 1e-2:
                return label

    raise ConfigError(
        f"Invalid pressure label {value!r}. Use a subset of {list(PRESSURE_TO_PA)}."
    )


def normalize_pressures(pressures: Sequence[object]) -> List[str]:
    """Normalize and deduplicate pressure labels while preserving user order."""
    normalized: List[str] = []
    seen = set()
    for pressure in pressures:
        label = normalize_pressure_label(pressure)
        if label in seen:
            continue
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


def find_flat_dataset_file() -> Path:
    """Find the flat rebuilt dataset file."""
    direct = DATA_SEARCH_ROOT / DATA_FLAT_FILE
    if direct.exists():
        return direct.resolve()

    matches = list(DATA_SEARCH_ROOT.glob("**/O2_simple_uniform_seed10_rebuilt_4p.txt"))
    if matches:
        # Prefer the shortest path if there are several copies.
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
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScalerNP":
        return cls(mean_=np.array(d["mean"], dtype=np.float64), scale_=np.array(d["scale"], dtype=np.float64))


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
            min_value = float(np.min(x))
            raise ValueError(f"Cannot log10-transform {name}: found non-positive value {min_value}.")
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

        return cls(
            log10_x=log10_x,
            log10_y=log10_y,
            standardize_x=standardize_x,
            standardize_y=standardize_y,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )

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
# Dataset loading and grouping
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

    df = pd.DataFrame(arr, columns=FLAT_COLUMNS)
    return df


def pressure_label_from_pa(pa: float, tolerance: float = 1e-2) -> str:
    for label, expected_pa in PRESSURE_TO_PA.items():
        if abs(pa - expected_pa) <= tolerance:
            return label
    raise ValueError(f"Unknown pressure value in dataset: {pa}")


def build_grouped_dataset(flat_path: Path, pressures: Sequence[object]) -> GroupedDataset:
    pressures_norm = normalize_pressures(pressures)
    df = load_flat_table(flat_path)

    df = df.copy()
    df["pressure_label"] = [pressure_label_from_pa(float(v)) for v in df["pressure_Pa"].to_numpy()]

    available_pressures = sorted(df["pressure_label"].unique(), key=lambda p: PRESSURE_TO_PA[p])
    missing = [p for p in pressures_norm if p not in available_pressures]
    if missing:
        raise ConfigError(
            f"Requested pressures {missing} are not available in dataset. "
            f"Available: {available_pressures}"
        )

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

    # Validate that K rows are aligned by sample index across all pressure blocks.
    ref_k = by_pressure[pressures_norm[0]][K_NAMES].to_numpy(dtype=np.float64)
    for p in pressures_norm[1:]:
        this_k = by_pressure[p][K_NAMES].to_numpy(dtype=np.float64)
        if not np.allclose(ref_k, this_k, rtol=0.0, atol=0.0):
            max_diff = float(np.max(np.abs(ref_k - this_k)))
            raise ValueError(
                "K values are not exactly aligned across pressure blocks. "
                f"Pressure {p}, max abs diff = {max_diff}."
            )

    x_parts = []
    feature_names = []
    for p in pressures_norm:
        densities = by_pressure[p][SPECIES].to_numpy(dtype=np.float64)
        x_parts.append(densities)
        for species in SPECIES:
            feature_names.append(f"{species}@{p}Torr")

    x_raw = np.concatenate(x_parts, axis=1)
    y_raw = ref_k.copy()
    sample_ids = np.arange(n_per_pressure, dtype=np.int64)

    if len(pressures_norm) < 3:
        print(
            "WARNING: fewer than 3 pressures selected. "
            "This is likely underdetermined/weak for 9 K targets and is useful mainly as a bad-answer test."
        )

    return GroupedDataset(
        pressures=pressures_norm,
        x_raw=x_raw,
        y_raw=y_raw,
        sample_ids=sample_ids,
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
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    raise ConfigError(f"Unknown activation {name!r}.")


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ======================================================================================
# Paths, compatibility, and metrics
# ======================================================================================


def saved_scheme_root() -> Path:
    return SAVE_WEIGHTS_ROOT / SCHEME


def saved_pressure_root(pressures: Sequence[str]) -> Path:
    return saved_scheme_root() / pressures_folder_name(pressures)


def saved_model_dir(pressures: Sequence[str], hidden_size: Sequence[int], seed: int) -> Path:
    return saved_pressure_root(pressures) / arch_folder_name(hidden_size) / f"seed_{int(seed)}"


def make_config_signature(
    dataset: GroupedDataset,
    hidden_size: Sequence[int],
    seed: int,
) -> Tuple[dict, str]:
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
    return config, stable_json_hash(config)


def is_compatible_saved_model(model_dir: Path, expected_signature: str) -> bool:
    info_path = model_dir / "model_info.json"
    model_path = model_dir / "model.pth"
    scalers_path = model_dir / "scalers.json"
    if not (info_path.exists() and model_path.exists() and scalers_path.exists()):
        return False
    try:
        info = read_json(info_path)
    except Exception:
        return False
    return info.get("config_signature") == expected_signature


def compute_metrics(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
) -> dict:
    err_scaled = y_pred_scaled - y_true_scaled
    mse_scaled = float(np.mean(err_scaled**2))
    rmse_scaled = float(np.sqrt(mse_scaled))

    y_true_log10 = np.log10(y_true_raw)
    y_pred_log10 = np.log10(y_pred_raw)
    err_log10 = y_pred_log10 - y_true_log10
    mse_log10 = float(np.mean(err_log10**2))
    rmse_log10 = float(np.sqrt(mse_log10))

    rel = np.abs((y_pred_raw - y_true_raw) / y_true_raw)
    mean_rel_per_k = rel.mean(axis=0) * 100.0
    max_rel_per_k = rel.max(axis=0) * 100.0

    metrics = {
        "test_mse_scaled": mse_scaled,
        "test_rmse_scaled": rmse_scaled,
        "test_mse_log10": mse_log10,
        "test_rmse_log10": rmse_log10,
        "mean_relative_error_percent": float(mean_rel_per_k.mean()),
        "max_relative_error_percent": float(max_rel_per_k.max()),
        "mean_relative_error_percent_per_k": {
            name: float(value) for name, value in zip(K_NAMES, mean_rel_per_k)
        },
        "max_relative_error_percent_per_k": {
            name: float(value) for name, value in zip(K_NAMES, max_rel_per_k)
        },
    }
    return metrics


# ======================================================================================
# Training
# ======================================================================================


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(x_tensor, y_tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


def evaluate_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    if len(x) == 0:
        return float("nan")

    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        pred = model(x_tensor)
        loss = criterion(pred, y_tensor)
    return float(loss.item())


def predict_scaled(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            batch = torch.tensor(x[start : start + 1024], dtype=torch.float32, device=device)
            pred = model(batch).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def train_one_model(
    dataset: GroupedDataset,
    hidden_size: Sequence[int],
    seed: int,
    device: torch.device,
) -> dict:
    model_dir = saved_model_dir(dataset.pressures, hidden_size, seed)
    model_dir.mkdir(parents=True, exist_ok=True)

    config, config_signature = make_config_signature(dataset, hidden_size, seed)

    if FORCE_RETRAIN and model_dir.exists():
        shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

    if not FORCE_RETRAIN and is_compatible_saved_model(model_dir, config_signature):
        info = read_json(model_dir / "model_info.json")
        metrics = read_json(model_dir / "metrics.json") if (model_dir / "metrics.json").exists() else {}
        return {
            "status": "reused",
            "model_dir": str(model_dir),
            "info": info,
            "metrics": metrics,
        }

    set_global_seed(seed)

    train_idx, val_idx, test_idx = split_indices(dataset.n_samples, seed, TEST_SPLIT, VAL_SPLIT)

    x_train_raw = dataset.x_raw[train_idx]
    y_train_raw = dataset.y_raw[train_idx]
    x_val_raw = dataset.x_raw[val_idx]
    y_val_raw = dataset.y_raw[val_idx]
    x_test_raw = dataset.x_raw[test_idx]
    y_test_raw = dataset.y_raw[test_idx]

    preprocessor = Preprocessor.fit(
        x_train_raw,
        y_train_raw,
        log10_x=LOG10_X,
        log10_y=LOG10_Y,
        standardize_x=STANDARDIZE_X,
        standardize_y=STANDARDIZE_Y,
    )

    x_train = preprocessor.transform_x(x_train_raw)
    y_train = preprocessor.transform_y(y_train_raw)
    x_val = preprocessor.transform_x(x_val_raw)
    y_val = preprocessor.transform_y(y_val_raw)
    x_test = preprocessor.transform_x(x_test_raw)
    y_test = preprocessor.transform_y(y_test_raw)

    model = MLP(
        input_size=dataset.input_size,
        output_size=dataset.output_size,
        hidden_size=hidden_size,
        activation=ACTIVATION,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = make_loader(x_train, y_train, BATCH_SIZE, shuffle=True, seed=seed)

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    loss_history = []

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_loss = evaluate_loss(model, x_val, y_val, device) if len(val_idx) else train_loss

        loss_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

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
    epochs_ran = len(loss_history)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train_loss = evaluate_loss(model, x_train, y_train, device)
    final_val_loss = evaluate_loss(model, x_val, y_val, device) if len(val_idx) else float("nan")

    y_test_pred_scaled = predict_scaled(model, x_test, device)
    y_test_pred_raw = preprocessor.inverse_transform_y(y_test_pred_scaled)

    metrics = compute_metrics(
        y_true_raw=y_test_raw,
        y_pred_raw=y_test_pred_raw,
        y_true_scaled=y_test,
        y_pred_scaled=y_test_pred_scaled,
    )

    info = {
        "script": Path(__file__).name,
        "scheme": SCHEME,
        "config": config,
        "config_signature": config_signature,
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
        },
        "training": {
            "seed": int(seed),
            "learning_rate": float(LEARNING_RATE),
            "batch_size": int(BATCH_SIZE),
            "max_epochs": int(MAX_EPOCHS),
            "patience": int(PATIENCE),
            "epochs_ran": int(epochs_ran),
            "best_epoch": int(best_epoch),
            "best_train_loss": float(best_train_loss),
            "best_val_loss": float(best_val_loss),
            "final_train_loss": float(final_train_loss),
            "final_val_loss": float(final_val_loss),
            "training_time_s": float(training_time_s),
            "device": str(device),
        },
        "metrics": metrics,
        "files": {
            "model_pth": str(model_dir / "model.pth"),
            "model_info_json": str(model_dir / "model_info.json"),
            "scalers_json": str(model_dir / "scalers.json"),
            "loss_history_csv": str(model_dir / "loss_history.csv"),
            "metrics_json": str(model_dir / "metrics.json"),
            "predictions_npz": str(model_dir / "test_predictions.npz"),
            "split_indices_npz": str(model_dir / "split_indices.npz"),
        },
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": info["model"],
        "preprocessing": preprocessor.to_dict(),
        "feature_names": dataset.feature_names,
        "target_names": dataset.target_names,
        "pressures": dataset.pressures,
        "config_signature": config_signature,
    }

    torch.save(checkpoint, model_dir / "model.pth")
    save_json(model_dir / "model_info.json", info)
    save_json(model_dir / "scalers.json", preprocessor.to_dict())
    save_json(model_dir / "metrics.json", metrics)

    pd.DataFrame(loss_history).to_csv(model_dir / "loss_history.csv", index=False)

    np.savez_compressed(
        model_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    np.savez_compressed(
        model_dir / "test_predictions.npz",
        y_test_true_raw=y_test_raw,
        y_test_pred_raw=y_test_pred_raw,
        y_test_true_scaled=y_test,
        y_test_pred_scaled=y_test_pred_scaled,
        test_idx=test_idx,
    )

    return {
        "status": "trained",
        "model_dir": str(model_dir),
        "info": info,
        "metrics": metrics,
    }


# ======================================================================================
# Summary files
# ======================================================================================


def make_summary_row(result: dict, dataset: GroupedDataset, hidden_size: Sequence[int], seed: int) -> dict:
    info = result.get("info", {})
    metrics = result.get("metrics", {}) or info.get("metrics", {}) or {}
    training = info.get("training", {})
    split = info.get("split", {})
    model = info.get("model", {})

    return {
        "scheme": SCHEME,
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
        "status": result.get("status"),
        "model_dir": result.get("model_dir"),
        "num_parameters": model.get("num_parameters"),
        "n_train": split.get("n_train"),
        "n_val": split.get("n_val"),
        "n_test": split.get("n_test"),
        "epochs_ran": training.get("epochs_ran"),
        "best_epoch": training.get("best_epoch"),
        "best_val_loss": training.get("best_val_loss"),
        "final_train_loss": training.get("final_train_loss"),
        "final_val_loss": training.get("final_val_loss"),
        "training_time_s": training.get("training_time_s", 0.0 if result.get("status") == "reused" else None),
        "test_mse_scaled": metrics.get("test_mse_scaled"),
        "test_rmse_scaled": metrics.get("test_rmse_scaled"),
        "test_mse_log10": metrics.get("test_mse_log10"),
        "test_rmse_log10": metrics.get("test_rmse_log10"),
        "mean_relative_error_percent": metrics.get("mean_relative_error_percent"),
        "max_relative_error_percent": metrics.get("max_relative_error_percent"),
    }


def save_summary_tables(rows: List[dict]) -> None:
    root = saved_scheme_root()
    root.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    summary_path = root / "pretrain_summary.csv"
    aggregate_path = root / "pretrain_aggregate_summary.csv"
    info_path = root / "pretrain_info.json"

    df.to_csv(summary_path, index=False)

    if not df.empty:
        aggregate = (
            df.groupby(["scheme", "pressures", "num_pressures", "arch_folder"], as_index=False)
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
                total_training_time_s=("training_time_s", "sum"),
            )
        )
        aggregate.to_csv(aggregate_path, index=False)

    run_info = {
        "script": Path(__file__).name,
        "purpose": "Train NN saved weights for the 9-K inverse-problem O2_simple dataset.",
        "scheme": SCHEME,
        "saved_weights_root": str(SAVE_WEIGHTS_ROOT),
        "scheme_saved_weights_root": str(root),
        "pressure_configs": [normalize_pressures(p) for p in PRESSURE_CONFIGS],
        "architectures": [list(a) for a in ARCHITECTURES],
        "seeds": SEEDS,
        "activation": ACTIVATION,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "test_split": TEST_SPLIT,
        "val_split": VAL_SPLIT,
        "log10_x": LOG10_X,
        "log10_y": LOG10_Y,
        "standardize_x": STANDARDIZE_X,
        "standardize_y": STANDARDIZE_Y,
        "force_retrain": FORCE_RETRAIN,
        "summary_csv": str(summary_path),
        "aggregate_csv": str(aggregate_path),
    }
    save_json(info_path, run_info)


# ======================================================================================
# Main workflow
# ======================================================================================


def main() -> None:
    flat_path = find_flat_dataset_file()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pressure_configs = [normalize_pressures(p) for p in PRESSURE_CONFIGS]

    SAVE_WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    saved_scheme_root().mkdir(parents=True, exist_ok=True)

    print(f"Dataset file: {flat_path}")
    print(f"Saved-weights root: {saved_scheme_root().resolve()}")
    print(f"Device: {device}")
    print(f"Pressure configs: {pressure_configs}")
    print(f"Architectures: {ARCHITECTURES}")
    print(f"Seeds: {SEEDS[0]} to {SEEDS[-1]} ({len(SEEDS)} seeds)")
    print(f"FORCE_RETRAIN: {FORCE_RETRAIN}")

    planned_models = len(pressure_configs) * len(ARCHITECTURES) * len(SEEDS)
    if MAX_MODELS_TO_TRAIN is not None:
        planned_display = min(planned_models, MAX_MODELS_TO_TRAIN)
    else:
        planned_display = planned_models
    print(f"Planned models this run: {planned_display} / {planned_models}")

    rows: List[dict] = []
    models_done = 0

    with tqdm(total=planned_display, desc="Training saved_weights") as pbar:
        for pressures in pressure_configs:
            dataset = build_grouped_dataset(flat_path, pressures)

            print("")
            print(f"Pressure config: {pressures}")
            print(f"Grouped inverse samples: {dataset.n_samples}")
            print(f"X shape: {dataset.x_raw.shape}")
            print(f"y shape: {dataset.y_raw.shape}")
            print(f"Feature names: {dataset.feature_names}")

            for hidden_size in ARCHITECTURES:
                for seed in SEEDS:
                    if MAX_MODELS_TO_TRAIN is not None and models_done >= MAX_MODELS_TO_TRAIN:
                        break

                    result = train_one_model(dataset, hidden_size, seed, device)
                    row = make_summary_row(result, dataset, hidden_size, seed)
                    rows.append(row)

                    status = result.get("status", "?")
                    rel = row.get("mean_relative_error_percent")
                    rel_text = "nan" if rel is None or pd.isna(rel) else f"{rel:.3g}%"

                    pbar.set_postfix(
                        pressures="_".join(pressures),
                        arch="_".join(map(str, hidden_size)),
                        seed=seed,
                        status=status,
                        rel=rel_text,
                    )
                    pbar.update(1)
                    models_done += 1

                if MAX_MODELS_TO_TRAIN is not None and models_done >= MAX_MODELS_TO_TRAIN:
                    break
            if MAX_MODELS_TO_TRAIN is not None and models_done >= MAX_MODELS_TO_TRAIN:
                break

    if SAVE_SUMMARY_FILES:
        save_summary_tables(rows)

    num_trained = sum(1 for r in rows if r.get("status") == "trained")
    num_reused = sum(1 for r in rows if r.get("status") == "reused")

    print("")
    print("Finished saved_weights training.")
    print(f"Models checked: {len(rows)}")
    print(f"Newly trained models: {num_trained}")
    print(f"Reused compatible models: {num_reused}")
    print(f"Saved weights folder: {saved_scheme_root().resolve()}")
    if SAVE_SUMMARY_FILES:
        print(f"Summary CSV: {(saved_scheme_root() / 'pretrain_summary.csv').resolve()}")
        print(f"Aggregate CSV: {(saved_scheme_root() / 'pretrain_aggregate_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
