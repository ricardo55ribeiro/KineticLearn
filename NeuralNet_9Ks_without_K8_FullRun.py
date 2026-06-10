from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
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

SCHEME = "O2_simple_9K_without_K8"
EXPERIMENT_NAME = "RunDefault9K_without_K8"

PRESSURE_CONFIGS = [["1", "2", "5", "10"]]
SPECIES = ["O2(X)", "O2(a)", "O(3P)"]

ALL_K_NAMES = [f"K{i}" for i in range(1, 10)]
TARGET_K_NAMES = ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K9"]
K_NAMES = TARGET_K_NAMES.copy()

K_REACTIONS_BY_NAME = {
    "K1": "e + O2(X) -> e + O2(a)",
    "K2": "e + O2(a) -> e + O2(X)",
    "K3": "e + O2(X) -> e + 2O(3P)",
    "K4": "e + O2(a) -> e + 2O(3P)",
    "K5": "O2(a) + O(3P) -> O2(X) + O(3P)",
    "K6": "O2(a) + O(3P) + O2(X) -> O2(X) + O(3P) + O2(X)",
    "K7": "2O(3P) + O2(X) -> 2O2(X)",
    "K8": "O2(a) + wall -> O2(X)",
    "K9": "O(3P) + wall -> 0.5O2(X)",
}
K_REACTIONS = [K_REACTIONS_BY_NAME[name] for name in K_NAMES]

ARCHITECTURES = [(30, 30), (30, 30, 30)]
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

BASE_RESULTS_DIR = Path("Results_NN")
SAVE_WEIGHTS_ROOT = Path("saved_weights")

# False is the normal mode: reuse compatible cached weights and regenerate Results_NN artifacts.
FORCE_RETRAIN = False

# If FORCE_RETRAIN is True, this creates a new Setup_XXX instead of overwriting old weights/results.
FORCE_RETRAIN_CREATES_NEW_SETUP = True

# Safety: do not overwrite a non-empty but incomplete/incompatible saved_weights model folder by default.
ALLOW_REPAIR_INCOMPLETE_CACHE = False

SAVE_SUMMARY_FILES = True
DATA_SEARCH_ROOT = Path(".")
DATA_FLAT_FILE = Path("9Ks_Dataset") / "O2_simple" / "O2_simple_uniform_generated_2000_4p.txt"
# Alternative examples:
# DATA_FLAT_FILE = Path("9Ks_Dataset") / "O2_simple" / "O2_simple_uniform_seed10_rebuilt_4p.txt"
# DATA_FLAT_FILE = Path("9Ks_Dataset") / "O2_simple" / "O2_simple_uniform_merged_2200_4p.txt"

MAX_MODELS_TO_TRAIN: Optional[int] = None
VERBOSE_EPOCH_LOSSES = False
DETERMINISTIC_TORCH = True

SAVE_PLOTS = True
SAVE_PREDICTIONS_CSV = True
SAVE_METRICS_TXT = True
LOSS_SMOOTHING_WINDOW = 25

PRESSURE_TO_PA = {
    "1": 133.33,
    "2": 266.66,
    "5": 666.66,
    "10": 1333.30,
}

FLAT_COLUMNS = ALL_K_NAMES + ["pressure_Pa"] + SPECIES


class ConfigError(ValueError):
    pass


# ======================================================================================
# Basic utilities
# ======================================================================================


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
    normalized, seen = [], set()
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


def setup_folder_name(number: int) -> str:
    return f"Setup_{int(number):03d}"


def setup_number_from_name(name: str) -> Optional[int]:
    match = re.fullmatch(r"Setup_(\d{3,})", name)
    return int(match.group(1)) if match else None


def canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    # Dataset content identity only; never used in folder names.
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def find_flat_dataset_file() -> Path:
    direct = DATA_SEARCH_ROOT / DATA_FLAT_FILE
    if direct.exists():
        return direct.resolve()

    filename = DATA_FLAT_FILE.name
    matches = list(DATA_SEARCH_ROOT.glob(f"**/{filename}"))
    if matches:
        matches.sort(key=lambda p: (len(str(p)), str(p)))
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not find {filename}.\n"
        f"Looked under: {DATA_SEARCH_ROOT.resolve()}\n"
        f"Expected path: {direct}\n"
        "Place the generated/extracted dataset folder in the project root, or update DATA_FLAT_FILE."
    )


# ======================================================================================
# Human-readable Setup registry
# ======================================================================================


def saved_scheme_root() -> Path:
    return SAVE_WEIGHTS_ROOT / SCHEME


def saved_setup_root(setup_name: str) -> Path:
    return saved_scheme_root() / setup_name


def saved_model_dir(setup_name: str, pressures: Sequence[str], hidden_size: Sequence[int], seed: int) -> Path:
    return saved_setup_root(setup_name) / pressures_folder_name(pressures) / f"seed_{int(seed):04d}" / arch_folder_name(hidden_size)


def results_experiment_base_root() -> Path:
    return BASE_RESULTS_DIR / SCHEME / EXPERIMENT_NAME


def results_setup_root(setup_name: str) -> Path:
    return results_experiment_base_root() / setup_name


def results_model_dir(setup_name: str, pressures: Sequence[str], hidden_size: Sequence[int], seed: int) -> Path:
    return results_setup_root(setup_name) / pressures_folder_name(pressures) / f"seed_{int(seed):04d}" / arch_folder_name(hidden_size)


def build_setup_settings(flat_path: Path, flat_sha256: str, pressure_configs: Sequence[Sequence[str]]) -> dict:
    return {
        "dataset": {
            "flat_dataset_name": flat_path.name,
            "configured_relative_path": str(DATA_FLAT_FILE),
            "resolved_path_at_creation": str(flat_path),
            "flat_dataset_sha256": flat_sha256,
            "flat_columns": FLAT_COLUMNS,
        },
        "problem": {
            "scheme": SCHEME,
            "experiment_name": EXPERIMENT_NAME,
            "pressure_configs": [normalize_pressures(p) for p in pressure_configs],
            "species": SPECIES,
            "all_k_names_in_flat_file": ALL_K_NAMES,
            "target_k_names": K_NAMES,
            "excluded_k_names": [name for name in ALL_K_NAMES if name not in K_NAMES],
            "target_k_reactions": {name: K_REACTIONS_BY_NAME[name] for name in K_NAMES},
        },
        "training": {
            "activation": ACTIVATION,
            "learning_rate": float(LEARNING_RATE),
            "batch_size": int(BATCH_SIZE),
            "max_epochs": int(MAX_EPOCHS),
            "patience": int(PATIENCE),
            "test_split": float(TEST_SPLIT),
            "val_split": float(VAL_SPLIT),
            "val_split_note": "VAL_SPLIT is applied to the remaining train+validation pool after removing TEST_SPLIT.",
        },
        "preprocessing": {
            "log10_x": bool(LOG10_X),
            "log10_y": bool(LOG10_Y),
            "standardize_x": bool(STANDARDIZE_X),
            "standardize_y": bool(STANDARDIZE_Y),
        },
    }


def setup_settings_text(settings: dict, setup_name: str) -> str:
    p = settings["problem"]
    t = settings["training"]
    d = settings["dataset"]
    pr = settings["preprocessing"]
    lines = [
        setup_name,
        "",
        "This setup defines the reusable data/training/preprocessing configuration.",
        "Architectures and seeds are intentionally not part of the setup identity.",
        "They are execution dimensions stored inside this setup.",
        "",
        "Dataset:",
        f"  file name: {d['flat_dataset_name']}",
        f"  configured relative path: {d['configured_relative_path']}",
        f"  resolved path when setup was created: {d['resolved_path_at_creation']}",
        f"  sha256: {d['flat_dataset_sha256']}",
        f"  flat columns: {d['flat_columns']}",
        "",
        "Problem:",
        f"  scheme: {p['scheme']}",
        f"  experiment name: {p['experiment_name']}",
        f"  pressure configs: {p['pressure_configs']}",
        f"  species: {p['species']}",
        f"  all K names in flat file: {p['all_k_names_in_flat_file']}",
        f"  network target K names: {p['target_k_names']}",
        f"  excluded K names: {p['excluded_k_names']}",
        "",
        "Training:",
        f"  activation: {t['activation']}",
        f"  learning rate: {t['learning_rate']}",
        f"  batch size: {t['batch_size']}",
        f"  max epochs: {t['max_epochs']}",
        f"  patience: {t['patience']}",
        f"  test split: {t['test_split']}",
        f"  val split: {t['val_split']}",
        f"  val split note: {t['val_split_note']}",
        "",
        "Preprocessing:",
        f"  log10 X: {pr['log10_x']}",
        f"  log10 y: {pr['log10_y']}",
        f"  standardize X: {pr['standardize_x']}",
        f"  standardize y: {pr['standardize_y']}",
        "",
        "Not part of setup identity:",
        "  architectures, seeds, max_models_to_train, plot/CSV/TXT artifact flags, verbose flags.",
    ]
    if "forced_retrain" in settings:
        lines.extend(["", f"Forced retrain metadata: {settings['forced_retrain']}"])
    return "\n".join(lines)


def write_setup_files(root: Path, setup_name: str, settings: dict) -> None:
    setup_root = root / setup_name
    setup_root.mkdir(parents=True, exist_ok=True)
    save_json(setup_root / "setup_settings.json", settings)
    with (setup_root / "setup_settings.txt").open("w", encoding="utf-8") as f:
        f.write(setup_settings_text(settings, setup_name))


def read_setup_settings(setup_root: Path) -> Optional[dict]:
    path = setup_root / "setup_settings.json"
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def build_setup_index_row(setup_name: str, settings: dict, created_at: str) -> dict:
    p = settings["problem"]
    t = settings["training"]
    d = settings["dataset"]
    return {
        "setup_name": setup_name,
        "created_at": created_at,
        "scheme": p["scheme"],
        "experiment_name": p["experiment_name"],
        "dataset_name": d["flat_dataset_name"],
        "dataset_sha256": d["flat_dataset_sha256"],
        "pressure_configs": json.dumps(p["pressure_configs"]),
        "species": json.dumps(p["species"]),
        "target_k_names": json.dumps(p["target_k_names"]),
        "excluded_k_names": json.dumps(p["excluded_k_names"]),
        "activation": t["activation"],
        "learning_rate": t["learning_rate"],
        "batch_size": t["batch_size"],
        "max_epochs": t["max_epochs"],
        "patience": t["patience"],
        "test_split": t["test_split"],
        "val_split": t["val_split"],
        "settings_json": canonical_json(settings),
    }


def collect_setup_index_entries(root: Path) -> List[dict]:
    root.mkdir(parents=True, exist_ok=True)
    entries: Dict[str, dict] = {}

    csv_path = root / "setups_index.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                setup_name = str(row.get("setup_name", "")).strip()
                if setup_name:
                    entries[setup_name] = row.to_dict()
        except Exception:
            pass

    for setup_root in sorted(root.glob("Setup_*")):
        if not setup_root.is_dir():
            continue
        settings = read_setup_settings(setup_root)
        if settings is None:
            continue
        old = entries.get(setup_root.name, {})
        created_at = old.get("created_at")
        if not created_at or pd.isna(created_at):
            created_at = datetime.fromtimestamp(setup_root.stat().st_ctime).isoformat(timespec="seconds")
        entries[setup_root.name] = build_setup_index_row(setup_root.name, settings, str(created_at))

    return [entries[name] for name in sorted(entries, key=lambda n: setup_number_from_name(n) or 10**9)]


def write_setups_index(root: Path, entries: List[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if entries:
        df = pd.DataFrame(entries).drop_duplicates(subset=["setup_name"], keep="last")
        df = df.sort_values(by="setup_name", key=lambda col: col.map(lambda x: setup_number_from_name(str(x)) or 10**9))
    else:
        df = pd.DataFrame()
    df.to_csv(root / "setups_index.csv", index=False)

    lines = [
        "Setups index",
        "",
        "Each Setup_XXX is a human-readable configuration folder.",
        "A setup is reused only when setup_settings.json matches exactly.",
        "",
    ]
    if df.empty:
        lines.append("No setups registered yet.")
    else:
        for _, row in df.iterrows():
            lines += [
                str(row.get("setup_name", "")),
                f"  created_at: {row.get('created_at', '')}",
                f"  scheme: {row.get('scheme', '')}",
                f"  experiment_name: {row.get('experiment_name', '')}",
                f"  dataset_name: {row.get('dataset_name', '')}",
                f"  dataset_sha256: {row.get('dataset_sha256', '')}",
                f"  pressure_configs: {row.get('pressure_configs', '')}",
                f"  species: {row.get('species', '')}",
                f"  target_k_names: {row.get('target_k_names', '')}",
                f"  excluded_k_names: {row.get('excluded_k_names', '')}",
                f"  activation: {row.get('activation', '')}",
                f"  learning_rate: {row.get('learning_rate', '')}",
                f"  batch_size: {row.get('batch_size', '')}",
                f"  max_epochs: {row.get('max_epochs', '')}",
                f"  patience: {row.get('patience', '')}",
                f"  test_split: {row.get('test_split', '')}",
                f"  val_split: {row.get('val_split', '')}",
                "",
            ]
    with (root / "setups_index.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def next_available_setup_name(entries: List[dict], root: Path) -> str:
    used = set()
    for entry in entries:
        number = setup_number_from_name(str(entry.get("setup_name", "")))
        if number is not None:
            used.add(number)
    for setup_root in root.glob("Setup_*"):
        number = setup_number_from_name(setup_root.name)
        if number is not None:
            used.add(number)
    number = 1
    while number in used:
        number += 1
    return setup_folder_name(number)


def resolve_saved_setup(settings: dict, force_new_setup: bool = False) -> str:
    root = saved_scheme_root()
    root.mkdir(parents=True, exist_ok=True)
    entries = collect_setup_index_entries(root)
    settings_json = canonical_json(settings)

    if not force_new_setup:
        for entry in entries:
            if str(entry.get("settings_json", "")) == settings_json:
                setup_name = str(entry["setup_name"])
                write_setup_files(root, setup_name, settings)
                write_setups_index(root, collect_setup_index_entries(root))
                return setup_name

    setup_name = next_available_setup_name(entries, root)
    created_at = datetime.now().isoformat(timespec="seconds")
    write_setup_files(root, setup_name, settings)
    entries = collect_setup_index_entries(root)
    entries = [entry for entry in entries if entry.get("setup_name") != setup_name]
    entries.append(build_setup_index_row(setup_name, settings, created_at))
    write_setups_index(root, entries)
    return setup_name


def ensure_results_setup(setup_name: str, settings: dict) -> None:
    root = results_experiment_base_root()
    root.mkdir(parents=True, exist_ok=True)
    setup_root = root / setup_name
    existing = read_setup_settings(setup_root) if setup_root.exists() else None
    if existing is not None and canonical_json(existing) != canonical_json(settings):
        raise ConfigError(
            f"Results folder {setup_root} already exists with different setup_settings.json. "
            "Refusing to reuse the same setup name for different settings."
        )
    write_setup_files(root, setup_name, settings)
    entries = collect_setup_index_entries(root)
    if setup_name not in {str(entry.get("setup_name", "")) for entry in entries}:
        entries.append(build_setup_index_row(setup_name, settings, datetime.now().isoformat(timespec="seconds")))
    write_setups_index(root, entries)


def append_invocation_log(setup_root: Path, setup_name: str) -> None:
    setup_root.mkdir(parents=True, exist_ok=True)
    path = setup_root / "run_invocation_log.csv"
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "setup_name": setup_name,
        "architectures_requested": json.dumps([list(map(int, a)) for a in ARCHITECTURES]),
        "seeds_requested": json.dumps([int(s) for s in SEEDS]),
        "max_models_to_train": MAX_MODELS_TO_TRAIN,
        "save_plots": bool(SAVE_PLOTS),
        "save_predictions_csv": bool(SAVE_PREDICTIONS_CSV),
        "save_metrics_txt": bool(SAVE_METRICS_TXT),
        "force_retrain": bool(FORCE_RETRAIN),
        "force_retrain_creates_new_setup": bool(FORCE_RETRAIN_CREATES_NEW_SETUP),
    }
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)


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
        return cls(
            log10_x=log10_x,
            log10_y=log10_y,
            standardize_x=standardize_x,
            standardize_y=standardize_y,
            x_scaler=StandardScalerNP.fit(x_work) if standardize_x else None,
            y_scaler=StandardScalerNP.fit(y_work) if standardize_y else None,
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
# Dataset loading
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

    ref_all_k = by_pressure[pressures_norm[0]][ALL_K_NAMES].to_numpy(dtype=np.float64)
    for p in pressures_norm[1:]:
        this_all_k = by_pressure[p][ALL_K_NAMES].to_numpy(dtype=np.float64)
        if not np.allclose(ref_all_k, this_all_k, rtol=0.0, atol=0.0):
            max_diff = float(np.max(np.abs(ref_all_k - this_all_k)))
            raise ValueError(f"K values are not aligned across pressure blocks. Pressure {p}, max abs diff = {max_diff}.")

    x_parts, feature_names = [], []
    for p in pressures_norm:
        x_parts.append(by_pressure[p][SPECIES].to_numpy(dtype=np.float64))
        for species in SPECIES:
            feature_names.append(f"{species}@{p}Torr")

    x_raw = np.concatenate(x_parts, axis=1)
    y_raw = by_pressure[pressures_norm[0]][K_NAMES].to_numpy(dtype=np.float64)
    sample_ids = np.arange(n_per_pressure, dtype=np.int64)

    if len(pressures_norm) < 3:
        print(
            "WARNING: fewer than 3 pressures selected. "
            f"This is likely underdetermined/weak for {len(K_NAMES)} selected K targets."
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
# Model and training
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


def make_model_cache_identity(
    setup_name: str,
    setup_settings: dict,
    dataset: GroupedDataset,
    hidden_size: Sequence[int],
    seed: int,
) -> dict:
    return {
        "setup_name": setup_name,
        "setup_settings": setup_settings,
        "pressures": dataset.pressures,
        "species": SPECIES,
        "target_names": dataset.target_names,
        "input_size": int(dataset.input_size),
        "output_size": int(dataset.output_size),
        "hidden_size": list(map(int, hidden_size)),
        "seed": int(seed),
    }


def cache_dir_has_any_files(weights_dir: Path) -> bool:
    return weights_dir.exists() and any(weights_dir.iterdir())


def is_compatible_saved_cache(weights_dir: Path, expected_cache_identity: dict) -> bool:
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
    return info.get("cache_identity") == expected_cache_identity


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator, drop_last=False)


def evaluate_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    if len(x) == 0:
        return float("nan")
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        return float(nn.MSELoss()(model(x_tensor), y_tensor).item())


def predict_scaled(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            batch = torch.tensor(x[start:start + 1024], dtype=torch.float32, device=device)
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_model_from_scratch(
    dataset: GroupedDataset,
    hidden_size: Sequence[int],
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, Preprocessor, dict, List[dict], np.ndarray, np.ndarray, np.ndarray]:
    set_global_seed(seed)
    train_idx, val_idx, test_idx = split_indices(dataset.n_samples, seed, TEST_SPLIT, VAL_SPLIT)

    x_train_raw = dataset.x_raw[train_idx]
    y_train_raw = dataset.y_raw[train_idx]
    x_val_raw = dataset.x_raw[val_idx]
    y_val_raw = dataset.y_raw[val_idx]

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
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_loss = evaluate_loss(model, x_val, y_val, device) if len(val_idx) else train_loss
        loss_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

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
    training_info = {
        "seed": int(seed),
        "learning_rate": float(LEARNING_RATE),
        "batch_size": int(BATCH_SIZE),
        "max_epochs": int(MAX_EPOCHS),
        "patience": int(PATIENCE),
        "epochs_ran": int(len(loss_history)),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "training_time_s": float(training_time_s),
        "device": str(device),
    }
    return model, preprocessor, training_info, loss_history, train_idx, val_idx, test_idx


def save_loss_history_csv(output_dir: Path, loss_history: List[dict]) -> None:
    if not loss_history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(loss_history)
    if "train_loss" in df.columns:
        df["train_loss_smooth"] = moving_average(df["train_loss"].to_numpy(), window=LOSS_SMOOTHING_WINDOW)
    if "val_loss" in df.columns:
        df["val_loss_smooth"] = moving_average(df["val_loss"].to_numpy(), window=LOSS_SMOOTHING_WINDOW)
    df.to_csv(output_dir / "loss_history.csv", index=False)


def read_loss_history_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if {"epoch", "train_loss", "val_loss"}.issubset(df.columns):
        return df[["epoch", "train_loss", "val_loss"]].to_dict(orient="records")
    return []


def save_cache_artifacts(
    weights_dir: Path,
    model: nn.Module,
    preprocessor: Preprocessor,
    cache_info: dict,
    loss_history: List[dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_dir / "model.pth")
    save_json(weights_dir / "model_cache_info.json", cache_info)
    save_json(weights_dir / "scalers.json", preprocessor.to_dict())
    save_loss_history_csv(weights_dir, loss_history)
    np.savez_compressed(weights_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def load_cached_model_and_artifacts(
    weights_dir: Path,
    dataset: GroupedDataset,
    hidden_size: Sequence[int],
    device: torch.device,
) -> Tuple[nn.Module, Preprocessor, dict, List[dict], np.ndarray, np.ndarray, np.ndarray]:
    cache_info = read_json(weights_dir / "model_cache_info.json")
    preprocessor = Preprocessor.from_dict(read_json(weights_dir / "scalers.json"))
    split = np.load(weights_dir / "split_indices.npz")
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
    loss_history = read_loss_history_csv(weights_dir / "loss_history.csv")

    model = MLP(dataset.input_size, dataset.output_size, hidden_size, ACTIVATION).to(device)
    loaded = torch.load(weights_dir / "model.pth", map_location=device)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        loaded = loaded["model_state_dict"]
    model.load_state_dict(loaded)
    model.eval()
    return model, preprocessor, cache_info, loss_history, train_idx, val_idx, test_idx


# ======================================================================================
# Results, plots, summaries
# ======================================================================================


def compute_metrics(y_true_raw, y_pred_raw, y_true_scaled, y_pred_scaled) -> dict:
    err_scaled = y_pred_scaled - y_true_scaled
    mse_scaled = float(np.mean(err_scaled ** 2))
    y_true_log10 = np.log10(y_true_raw)
    y_pred_log10 = np.log10(y_pred_raw)
    err_log10 = y_pred_log10 - y_true_log10
    mse_log10 = float(np.mean(err_log10 ** 2))
    rel = np.abs((y_pred_raw - y_true_raw) / y_true_raw)
    mean_rel_per_k = rel.mean(axis=0) * 100.0
    max_rel_per_k = rel.max(axis=0) * 100.0
    return {
        "test_mse_scaled": mse_scaled,
        "test_rmse_scaled": float(np.sqrt(mse_scaled)),
        "test_mse_log10": mse_log10,
        "test_rmse_log10": float(np.sqrt(mse_log10)),
        "mean_relative_error_percent": float(mean_rel_per_k.mean()),
        "max_relative_error_percent": float(max_rel_per_k.max()),
        "mean_relative_error_percent_per_k": {name: float(value) for name, value in zip(K_NAMES, mean_rel_per_k)},
        "max_relative_error_percent_per_k": {name: float(value) for name, value in zip(K_NAMES, max_rel_per_k)},
    }


def _set_matplotlib_backend() -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)


def save_predictions_csv(output_dir, y_true_raw, y_pred_raw, y_true_scaled, y_pred_scaled, test_idx) -> None:
    if not SAVE_PREDICTIONS_CSV:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    data: Dict[str, np.ndarray] = {"sample_id": np.asarray(test_idx, dtype=np.int64)}
    for i, name in enumerate(K_NAMES):
        true_raw = y_true_raw[:, i]
        pred_raw = y_pred_raw[:, i]
        true_scaled = y_true_scaled[:, i]
        pred_scaled = y_pred_scaled[:, i]
        denominator = np.where(np.abs(true_raw) < 1e-300, 1e-300, true_raw)
        rel_err = np.abs((pred_raw - true_raw) / denominator)
        data[f"{name}_true_scaled"] = true_scaled
        data[f"{name}_pred_scaled"] = pred_scaled
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
    model_info = info.get("model", {})
    split = info.get("split", {})
    training = info.get("training", {})
    dataset_info = info.get("dataset", {})
    lines = [
        f"Scheme: {info.get('scheme')}",
        f"Experiment name: {info.get('experiment_name')}",
        f"Setup: {info.get('setup_name')}",
        f"Status: {info.get('status')}",
        f"Saved weights path: {info.get('saved_weights_path')}",
        f"Results dir: {info.get('results_dir')}",
        f"Pressures: {dataset_info.get('pressures')}",
        f"Species: {dataset_info.get('species')}",
        f"Target K names: {dataset_info.get('target_names')}",
        f"Excluded K names: {dataset_info.get('excluded_k_names')}",
        f"Hidden size: {model_info.get('hidden_size')}",
        f"Activation: {model_info.get('activation')}",
        f"Input size: {model_info.get('input_size')}",
        f"Output size: {model_info.get('output_size')}",
        f"Number of parameters: {model_info.get('num_parameters')}",
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
        f"Cached/original training time (s): {training.get('training_time_s')}",
        f"Current invocation training time (s): {info.get('current_invocation_training_time_s')}",
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
    for name in K_NAMES:
        lines.append(f"  {name}: mean={mean_rel.get(name)} %, max={max_rel.get(name)} % | {K_REACTIONS_BY_NAME[name]}")
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_loss_curves(loss_history: List[dict], output_dir: Path, log_scale: bool = True) -> None:
    if not SAVE_PLOTS or not loss_history:
        return
    _set_matplotlib_backend()
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(loss_history)
    epochs = df["epoch"].to_numpy(dtype=float) if "epoch" in df.columns else np.arange(1, len(df) + 1, dtype=float)
    train_loss = df["train_loss"].to_numpy(dtype=float)
    val_loss = df["val_loss"].to_numpy(dtype=float)
    plt.rcParams.update({"font.size": 14, "text.usetex": False})
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, moving_average(train_loss), linewidth=1.8, label="Training Loss")
    plt.plot(epochs, moving_average(val_loss), linewidth=1.8, label="Validation Loss")
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


def plot_predictions_grid(output_dir, y_true, y_pred, *, filename_base, title_prefix, relative_error_true_denominator=True) -> None:
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
            vmin, vmax = float(np.min(finite)), float(np.max(finite))
            pad = max(abs(vmin) * 0.05, 1e-12) if abs(vmax - vmin) < 1e-15 else 0.05 * (vmax - vmin)
            vmin -= pad
            vmax += pad
            ax.plot([vmin, vmax], [vmin, vmax], "--", color="black", linewidth=1)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
        denom_src = true_i if relative_error_true_denominator else pred_i
        denom = np.where(np.abs(denom_src) < 1e-300, 1e-300, denom_src)
        rel_err = np.abs((pred_i - true_i) / denom)
        mean_rel = float(np.mean(rel_err) * 100.0) if rel_err.size else float("nan")
        max_rel = float(np.max(rel_err) * 100.0) if rel_err.size else float("nan")
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{title_prefix} {K_NAMES[i]}")
        ax.text(0.05, 0.95, f"Mean δrel={mean_rel:.2f}%\nMax δrel={max_rel:.2f}%", fontsize=10,
                transform=ax.transAxes, verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.25))
        if rel_err.size:
            max_index = int(np.argmax(rel_err))
            ax.scatter(true_i[max_index], pred_i[max_index], color="gold", edgecolor="black", zorder=3, s=45)
    for j in range(output_size, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename_base}.pdf")
    fig.savefig(output_dir / f"{filename_base}.png", dpi=200)
    plt.close(fig)


def plot_results(output_dir, y_true_raw, y_pred_raw, y_true_scaled, y_pred_scaled) -> None:
    if not SAVE_PLOTS:
        return
    plot_predictions_grid(output_dir, y_true_scaled, y_pred_scaled, filename_base="NeuralNet", title_prefix="Scaled")
    if np.all(y_true_raw > 0) and np.all(y_pred_raw > 0):
        plot_predictions_grid(
            output_dir,
            np.log10(y_true_raw),
            np.log10(y_pred_raw),
            filename_base="NeuralNet_log10_raw",
            title_prefix="log10 raw",
            relative_error_true_denominator=False,
        )


def save_result_artifacts(result_dir, info, metrics, loss_history, y_true_raw, y_pred_raw, y_true_scaled, y_pred_scaled, train_idx, val_idx, test_idx):
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_dir / "model_info.json", info)
    save_json(result_dir / "metrics.json", metrics)
    save_metrics_txt(result_dir, info, metrics)
    save_loss_history_csv(result_dir, loss_history)
    np.savez_compressed(result_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    np.savez_compressed(
        result_dir / "test_predictions.npz",
        y_test_true_raw=y_true_raw,
        y_test_pred_raw=y_pred_raw,
        y_test_true_scaled=y_true_scaled,
        y_test_pred_scaled=y_pred_scaled,
        test_idx=test_idx,
    )
    save_predictions_csv(result_dir, y_true_raw, y_pred_raw, y_true_scaled, y_pred_scaled, test_idx)
    plot_loss_curves(loss_history, result_dir, log_scale=True)
    plot_results(result_dir, y_true_raw, y_pred_raw, y_true_scaled, y_pred_scaled)


def evaluate_and_write_results(
    setup_name,
    setup_settings,
    dataset,
    hidden_size,
    seed,
    device,
    model,
    preprocessor,
    cache_info,
    loss_history,
    train_idx,
    val_idx,
    test_idx,
    weights_dir,
    result_dir,
    cache_identity,
    status,
) -> dict:
    x_test_raw = dataset.x_raw[test_idx]
    y_test_raw = dataset.y_raw[test_idx]
    y_test_scaled = preprocessor.transform_y(y_test_raw)
    x_test_scaled = preprocessor.transform_x(x_test_raw)
    y_pred_scaled = predict_scaled(model, x_test_scaled, device)
    y_pred_raw = preprocessor.inverse_transform_y(y_pred_scaled)
    metrics = compute_metrics(y_test_raw, y_pred_raw, y_test_scaled, y_pred_scaled)
    training_info = cache_info.get("training", {})
    current_training_time = 0.0 if status == "reused" else float(training_info.get("training_time_s", 0.0))

    info = {
        "script": Path(__file__).name,
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "setup_name": setup_name,
        "status": status,
        "saved_weights_path": str(weights_dir / "model.pth"),
        "saved_weights_dir": str(weights_dir),
        "results_dir": str(result_dir),
        "cache_identity": cache_identity,
        "setup_settings": setup_settings,
        "current_invocation_training_time_s": current_training_time,
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
            "all_k_names_in_flat_file": ALL_K_NAMES,
            "excluded_k_names": [name for name in ALL_K_NAMES if name not in K_NAMES],
            "target_k_reactions": {name: K_REACTIONS_BY_NAME[name] for name in K_NAMES},
            "all_k_reactions": K_REACTIONS_BY_NAME,
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
        "training": training_info,
        "metrics": metrics,
    }
    save_result_artifacts(result_dir, info, metrics, loss_history, y_test_raw, y_pred_raw, y_test_scaled, y_pred_scaled, train_idx, val_idx, test_idx)
    return {
        "status": status,
        "model_dir": str(result_dir),
        "results_dir": str(result_dir),
        "saved_weights_path": str(weights_dir / "model.pth"),
        "saved_weights_dir": str(weights_dir),
        "setup_name": setup_name,
        "info": info,
        "metrics": metrics,
    }


def get_or_train_and_write_results(setup_name, setup_settings, dataset, hidden_size, seed, device) -> dict:
    cache_identity = make_model_cache_identity(setup_name, setup_settings, dataset, hidden_size, seed)
    weights_dir = saved_model_dir(setup_name, dataset.pressures, hidden_size, seed)
    result_dir = results_model_dir(setup_name, dataset.pressures, hidden_size, seed)
    weights_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    compatible = is_compatible_saved_cache(weights_dir, cache_identity)
    if not FORCE_RETRAIN and compatible:
        model, preprocessor, cache_info, loss_history, train_idx, val_idx, test_idx = load_cached_model_and_artifacts(weights_dir, dataset, hidden_size, device)
        return evaluate_and_write_results(
            setup_name, setup_settings, dataset, hidden_size, seed, device, model, preprocessor, cache_info,
            loss_history, train_idx, val_idx, test_idx, weights_dir, result_dir, cache_identity, "reused"
        )

    if cache_dir_has_any_files(weights_dir) and not compatible and not ALLOW_REPAIR_INCOMPLETE_CACHE:
        raise ConfigError(
            "Found an existing saved_weights model folder, but it is incomplete or incompatible:\n"
            f"  {weights_dir}\n"
            "Refusing to overwrite it. Delete only that incomplete folder manually, or set "
            "ALLOW_REPAIR_INCOMPLETE_CACHE = True if you are sure you want to replace it."
        )

    model, preprocessor, training_info, loss_history, train_idx, val_idx, test_idx = train_model_from_scratch(dataset, hidden_size, seed, device)
    cache_info = {
        "script": Path(__file__).name,
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "setup_name": setup_name,
        "cache_identity": cache_identity,
        "setup_settings": setup_settings,
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
            "all_k_names_in_flat_file": ALL_K_NAMES,
            "excluded_k_names": [name for name in ALL_K_NAMES if name not in K_NAMES],
            "target_k_reactions": {name: K_REACTIONS_BY_NAME[name] for name in K_NAMES},
            "all_k_reactions": K_REACTIONS_BY_NAME,
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
        "training": training_info,
    }
    save_cache_artifacts(weights_dir, model, preprocessor, cache_info, loss_history, train_idx, val_idx, test_idx)
    return evaluate_and_write_results(
        setup_name, setup_settings, dataset, hidden_size, seed, device, model, preprocessor, cache_info,
        loss_history, train_idx, val_idx, test_idx, weights_dir, result_dir, cache_identity, "trained"
    )


def make_summary_row(result: dict, dataset: GroupedDataset, hidden_size: Sequence[int], seed: int) -> dict:
    info = result.get("info", {})
    metrics = result.get("metrics", {}) or info.get("metrics", {}) or {}
    training = info.get("training", {})
    split = info.get("split", {})
    model = info.get("model", {})
    row = {
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "setup_name": result.get("setup_name"),
        "pressures": ",".join(dataset.pressures),
        "pressure_folder": pressures_folder_name(dataset.pressures),
        "num_pressures": len(dataset.pressures),
        "species": ",".join(SPECIES),
        "target_names": ",".join(K_NAMES),
        "excluded_k_names": ",".join([name for name in ALL_K_NAMES if name not in K_NAMES]),
        "num_species": len(SPECIES),
        "num_grouped_samples": dataset.n_samples,
        "input_size": dataset.input_size,
        "output_size": dataset.output_size,
        "hidden_size": ",".join(map(str, hidden_size)),
        "arch_folder": arch_folder_name(hidden_size),
        "seed": int(seed),
        "status": result.get("status"),
        "results_dir": result.get("results_dir", result.get("model_dir")),
        "saved_weights_path": result.get("saved_weights_path"),
        "saved_weights_dir": result.get("saved_weights_dir"),
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
        "cached_training_time_s": training.get("training_time_s"),
        "current_invocation_training_time_s": info.get("current_invocation_training_time_s"),
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
        f"Architecture comparison summary for {SCHEME}",
        "",
        f"Scheme: {SCHEME}",
        f"Experiment name: {EXPERIMENT_NAME}",
        f"Target K names: {K_NAMES}",
        f"Excluded K names: {[name for name in ALL_K_NAMES if name not in K_NAMES]}",
        f"Architectures requested in latest script: {[list(a) for a in ARCHITECTURES]}",
        f"Seeds requested in latest script: {SEEDS}",
        f"Activation: {ACTIVATION}",
        f"Learning rate: {LEARNING_RATE}",
        f"Batch size: {BATCH_SIZE}",
        f"Max epochs: {MAX_EPOCHS}",
        f"Patience: {PATIENCE}",
        f"TEST_SPLIT: {TEST_SPLIT}",
        f"VAL_SPLIT: {VAL_SPLIT} (fraction of remaining train+validation pool)",
        "",
    ]
    for row in rows:
        lines += [
            f"Seed: {row.get('seed')}",
            f"Architecture: {row.get('hidden_size')}",
            f"  Setup: {row.get('setup_name')}",
            f"  Pressures: {row.get('pressures')}",
            f"  Status: {row.get('status')}",
            f"  Results dir: {row.get('results_dir')}",
            f"  Saved weights: {row.get('saved_weights_path')}",
            f"  Train/Val/Test: {row.get('n_train')}/{row.get('n_val')}/{row.get('n_test')}",
            f"  Epochs ran: {row.get('epochs_ran')}",
            f"  Best epoch: {row.get('best_epoch')}",
            f"  Best val loss: {row.get('best_val_loss')}",
            f"  Test MSE scaled: {row.get('test_mse_scaled')}",
            f"  Test RMSE log10: {row.get('test_rmse_log10')}",
            f"  Mean relative error (%): {row.get('mean_relative_error_percent')}",
            f"  Max relative error (%): {row.get('max_relative_error_percent')}",
            "",
        ]
    with (root / filename).open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_summary_tables(new_rows: List[dict], setup_name: str, setup_settings: dict) -> None:
    setup_root = results_setup_root(setup_name)
    setup_root.mkdir(parents=True, exist_ok=True)
    summary_path = setup_root / "pretrain_summary.csv"
    aggregate_path = setup_root / "pretrain_aggregate_summary.csv"
    info_path = setup_root / "pretrain_info.json"
    text_summary_path = setup_root / "summary.txt"

    new_df = pd.DataFrame(new_rows)
    if summary_path.exists():
        old_df = pd.read_csv(summary_path)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    if not df.empty:
        dedupe_cols = ["scheme", "experiment_name", "setup_name", "pressures", "arch_folder", "seed"]
        df = df.drop_duplicates(subset=[c for c in dedupe_cols if c in df.columns], keep="last")
    df.to_csv(summary_path, index=False)

    if not df.empty:
        aggregate = (
            df.groupby(["scheme", "experiment_name", "setup_name", "pressures", "num_pressures", "arch_folder"], as_index=False)
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
                total_current_invocation_training_time_s=("current_invocation_training_time_s", "sum"),
                mean_cached_training_time_s=("cached_training_time_s", "mean"),
            )
        )
        aggregate.to_csv(aggregate_path, index=False)
    else:
        pd.DataFrame().to_csv(aggregate_path, index=False)

    all_rows = df.to_dict(orient="records") if not df.empty else []
    save_global_summary_text(setup_root, all_rows, filename="summary.txt")

    save_json(info_path, {
        "script": Path(__file__).name,
        "purpose": "Train/reuse NN saved weights and regenerate Results_NN artifacts for the O2_simple inverse-problem dataset.",
        "scheme": SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "setup_name": setup_name,
        "setup_settings": setup_settings,
        "saved_weights_root": str(SAVE_WEIGHTS_ROOT),
        "scheme_saved_weights_root": str(saved_scheme_root()),
        "results_root": str(BASE_RESULTS_DIR),
        "setup_results_root": str(setup_root),
        "architectures_requested_latest_invocation": [list(a) for a in ARCHITECTURES],
        "seeds_requested_latest_invocation": SEEDS,
        "max_models_to_train_latest_invocation": MAX_MODELS_TO_TRAIN,
        "force_retrain_latest_invocation": FORCE_RETRAIN,
        "force_retrain_creates_new_setup": FORCE_RETRAIN_CREATES_NEW_SETUP,
        "save_plots_latest_invocation": SAVE_PLOTS,
        "save_predictions_csv_latest_invocation": SAVE_PREDICTIONS_CSV,
        "save_metrics_txt_latest_invocation": SAVE_METRICS_TXT,
        "summary_csv": str(summary_path),
        "aggregate_csv": str(aggregate_path),
        "summary_txt": str(text_summary_path),
        "num_summary_rows_total": int(len(df)),
    })


# ======================================================================================
# Main workflow
# ======================================================================================


def main() -> None:
    flat_path = find_flat_dataset_file()
    flat_sha256 = sha256_file(flat_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pressure_configs = [normalize_pressures(p) for p in PRESSURE_CONFIGS]

    setup_settings = build_setup_settings(flat_path, flat_sha256, pressure_configs)
    force_new_setup = bool(FORCE_RETRAIN and FORCE_RETRAIN_CREATES_NEW_SETUP)
    if force_new_setup:
        setup_settings = {
            **setup_settings,
            "forced_retrain": {
                "requested": True,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "note": "FORCE_RETRAIN was True, so a new setup was created instead of overwriting an existing setup.",
            },
        }

    SAVE_WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_scheme_root().mkdir(parents=True, exist_ok=True)
    results_experiment_base_root().mkdir(parents=True, exist_ok=True)

    setup_name = resolve_saved_setup(setup_settings, force_new_setup=force_new_setup)
    ensure_results_setup(setup_name, setup_settings)
    saved_setup = saved_setup_root(setup_name)
    results_setup = results_setup_root(setup_name)
    append_invocation_log(results_setup, setup_name)

    print(f"Dataset file: {flat_path}")
    print(f"Dataset sha256: {flat_sha256}")
    print(f"Saved-weights setup root: {saved_setup.resolve()}")
    print(f"Results setup root: {results_setup.resolve()}")
    print(f"Setup: {setup_name}")
    print(f"Device: {device}")
    print(f"Pressure configs: {pressure_configs}")
    print(f"Flat-file K columns: {ALL_K_NAMES}")
    print(f"Network target K columns: {K_NAMES}")
    print(f"Excluded K columns: {[name for name in ALL_K_NAMES if name not in K_NAMES]}")
    print(f"Architectures requested this invocation: {ARCHITECTURES}")
    print(f"Seeds requested this invocation: {SEEDS[0]} to {SEEDS[-1]} ({len(SEEDS)} seeds)")
    print(f"FORCE_RETRAIN: {FORCE_RETRAIN}")
    print(f"FORCE_RETRAIN_CREATES_NEW_SETUP: {FORCE_RETRAIN_CREATES_NEW_SETUP}")
    print(f"ALLOW_REPAIR_INCOMPLETE_CACHE: {ALLOW_REPAIR_INCOMPLETE_CACHE}")
    print(f"SAVE_PLOTS: {SAVE_PLOTS}")
    print(f"SAVE_PREDICTIONS_CSV: {SAVE_PREDICTIONS_CSV}")
    print(f"SAVE_METRICS_TXT: {SAVE_METRICS_TXT}")

    planned_models = len(pressure_configs) * len(ARCHITECTURES) * len(SEEDS)
    planned_display = min(planned_models, MAX_MODELS_TO_TRAIN) if MAX_MODELS_TO_TRAIN is not None else planned_models
    print(f"Planned models this invocation: {planned_display} / {planned_models}")

    rows: List[dict] = []
    models_done = 0
    with tqdm(total=planned_display, desc="Training/reusing saved_weights") as pbar:
        for pressures in pressure_configs:
            dataset = build_grouped_dataset(flat_path, pressures)
            print("")
            print(f"Pressure config: {pressures}")
            print(f"Grouped inverse samples: {dataset.n_samples}")
            print(f"X shape: {dataset.x_raw.shape}")
            print(f"y shape: {dataset.y_raw.shape}")
            print(f"Feature names: {dataset.feature_names}")
            print(f"Target names: {dataset.target_names}")

            for hidden_size in ARCHITECTURES:
                for seed in SEEDS:
                    if MAX_MODELS_TO_TRAIN is not None and models_done >= MAX_MODELS_TO_TRAIN:
                        break
                    result = get_or_train_and_write_results(setup_name, setup_settings, dataset, hidden_size, seed, device)
                    row = make_summary_row(result, dataset, hidden_size, seed)
                    rows.append(row)

                    rel = row.get("mean_relative_error_percent")
                    rel_text = "nan" if rel is None or pd.isna(rel) else f"{rel:.3g}%"
                    epochs = row.get("epochs_ran")
                    epochs_text = "?" if epochs is None or pd.isna(epochs) else str(int(epochs))
                    pbar.set_postfix(
                        setup=setup_name,
                        pressures="_".join(pressures),
                        arch="_".join(map(str, hidden_size)),
                        seed=seed,
                        status=result.get("status", "?"),
                        epochs=epochs_text,
                        rel=rel_text,
                    )
                    pbar.update(1)
                    models_done += 1
                if MAX_MODELS_TO_TRAIN is not None and models_done >= MAX_MODELS_TO_TRAIN:
                    break
            if MAX_MODELS_TO_TRAIN is not None and models_done >= MAX_MODELS_TO_TRAIN:
                break

    if SAVE_SUMMARY_FILES:
        save_summary_tables(rows, setup_name, setup_settings)

    num_trained = sum(1 for r in rows if r.get("status") == "trained")
    num_reused = sum(1 for r in rows if r.get("status") == "reused")

    print("")
    print("Finished saved_weights training/reuse and Results_NN artifact generation.")
    print(f"Models checked this invocation: {len(rows)}")
    print(f"Newly trained models this invocation: {num_trained}")
    print(f"Reused compatible models this invocation: {num_reused}")
    print(f"Saved weights setup folder: {saved_setup.resolve()}")
    print(f"Results setup folder: {results_setup.resolve()}")
    if SAVE_SUMMARY_FILES:
        print(f"Summary CSV: {(results_setup / 'pretrain_summary.csv').resolve()}")
        print(f"Aggregate CSV: {(results_setup / 'pretrain_aggregate_summary.csv').resolve()}")
        print(f"Summary TXT: {(results_setup / 'summary.txt').resolve()}")
        print(f"Setup settings TXT: {(results_setup / 'setup_settings.txt').resolve()}")
        print(f"Setups index CSV: {(results_experiment_base_root() / 'setups_index.csv').resolve()}")
        print(f"Setups index TXT: {(results_experiment_base_root() / 'setups_index.txt').resolve()}")


if __name__ == "__main__":
    main()
