from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ARCH_COLORS = {
    "30, 30": "blue",
    "30, 30, 30": "green",
    "50, 50": "red",
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def moving_average(x: Sequence[float], window: int = 25) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(x_padded, kernel, mode="valid")


def arch_to_folder_name(hidden_size: Sequence[int]) -> str:
    return ", ".join(str(h) for h in hidden_size)


def get_color_for_architecture(label: str) -> str:
    return ARCH_COLORS.get(label, "black")


def make_results_root(base_root: Path, scheme: str, experiment_name: str, add_timestamp: bool = True) -> Path:
    root = base_root / scheme / experiment_name
    if add_timestamp:
        root = root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root.mkdir(parents=True, exist_ok=True)
    return root


def prepare_results_folders(architectures: Sequence[Sequence[int]], root: Path) -> dict[tuple[int, ...], Path]:
    arch_dirs: dict[tuple[int, ...], Path] = {}
    for architecture in architectures:
        architecture = tuple(int(x) for x in architecture)
        arch_dir = root / arch_to_folder_name(architecture)
        arch_dir.mkdir(parents=True, exist_ok=True)
        arch_dirs[architecture] = arch_dir
    return arch_dirs


def save_json(filepath: Path, obj: Dict[str, Any]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=4)


def save_text(filepath: Path, text: str) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(text, encoding="utf-8")


def save_loss_history_csv(filepath: Path, history: dict[str, list[float]]) -> None:
    df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_loss_smooth": moving_average(history["train_loss"], window=25),
            "val_loss_smooth": moving_average(history["val_loss"], window=25),
        }
    )
    df.to_csv(filepath, index=False)


def plot_loss_curves(history: dict[str, list[float]], output_path: Path, log_scale: bool = True) -> None:
    plt.rcParams.update({"font.size": 14, "text.usetex": False})
    train_loss = np.asarray(history["train_loss"], dtype=float)
    val_loss = np.asarray(history["val_loss"], dtype=float)
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, moving_average(train_loss, window=25), linewidth=1.8, label="Training Loss")
    plt.plot(epochs, moving_average(val_loss, window=25), linewidth=1.8, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    if log_scale:
        plt.yscale("log")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_test_inputs_csv(output_path: Path, x_test_unscaled: np.ndarray, feature_names: Sequence[str]) -> None:
    df = pd.DataFrame(x_test_unscaled, columns=list(feature_names))
    df.insert(0, "sample_id", np.arange(len(df)))
    df.to_csv(output_path, index=False)


def metrics_to_text(metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in metrics.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def save_summary_csv(output_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    for column in df.columns:
        df[column] = df[column].apply(_stringify_if_needed)
    df.to_csv(output_path, index=False)


def save_global_summary_text(output_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    parts: list[str] = ["Summary", ""]
    for row in rows:
        first_key = next(iter(row.keys()), "row")
        parts.append(f"{first_key}: {row.get(first_key)}")
        for key, value in row.items():
            if key == first_key:
                continue
            parts.append(f"  {key}: {_stringify_if_needed(value)}")
        parts.append("")
    save_text(output_path, "\n".join(parts))


def _stringify_if_needed(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return value


def sanitize_yerr_for_log(y: np.ndarray, yerr: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    yerr = np.where(np.isnan(yerr), 0.0, yerr)
    yerr = np.where(yerr < 0.0, 0.0, yerr)
    return np.minimum(yerr, np.maximum(0.0, 0.999 * y))


def plot_architecture_curves(
    aggregated_rows: Sequence[dict[str, Any]] | pd.DataFrame,
    output_path: Path,
    y_col: str,
    yerr_col: str | None,
    title: str,
    y_label: str,
    yscale: str | None = None,
    architecture_order: Sequence[str] | None = None,
) -> None:
    df = aggregated_rows if isinstance(aggregated_rows, pd.DataFrame) else pd.DataFrame(aggregated_rows)
    if df.empty or y_col not in df.columns:
        return

    df = df.copy()
    df["observed_species_count"] = pd.to_numeric(df["observed_species_count"], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    if architecture_order is None:
        architecture_order = sorted(df["architecture"].dropna().unique().tolist())

    plt.rcParams.update({"font.size": 14, "text.usetex": False})
    plt.figure(figsize=(9, 6))

    plotted_any = False
    multiple_architectures = len(architecture_order) > 1

    for architecture_label in architecture_order:
        df_arch = df[df["architecture"] == architecture_label].copy()
        df_arch = df_arch.dropna(subset=["observed_species_count", y_col]).sort_values("observed_species_count")
        if df_arch.empty:
            continue

        x = df_arch["observed_species_count"].to_numpy(dtype=float)
        y = df_arch[y_col].to_numpy(dtype=float)

        yerr = None
        use_errorbars = False
        if yerr_col is not None and yerr_col in df_arch.columns:
            yerr_series = pd.to_numeric(df_arch[yerr_col], errors="coerce")
            if yerr_series.notna().any():
                yerr = yerr_series.to_numpy(dtype=float)
                yerr = np.where(np.isnan(yerr), 0.0, yerr)
                yerr = np.where(yerr < 0.0, 0.0, yerr)
                if yscale == "log":
                    yerr = sanitize_yerr_for_log(y, yerr)
                use_errorbars = np.any(yerr > 0)

        color = get_color_for_architecture(architecture_label)
        label = architecture_label if multiple_architectures else None

        if use_errorbars:
            plt.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.8,
                elinewidth=1.2,
                capsize=4,
                color=color,
                label=label,
            )
        else:
            plt.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                color=color,
                label=label,
            )

        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    plt.xlabel("Number of Observed Species")
    plt.ylabel(y_label)
    plt.title(title)

    if yscale == "log":
        positive = pd.to_numeric(df[y_col], errors="coerce").dropna()
        if not positive.empty and (positive > 0).all():
            plt.yscale("log")
    elif yscale == "symlog":
        plt.yscale("symlog", linthresh=1.0)

    xticks = sorted(pd.to_numeric(df["observed_species_count"], errors="coerce").dropna().unique().tolist())
    plt.xticks(xticks)
    plt.grid(True, which="both", alpha=0.3)

    if multiple_architectures:
        plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
