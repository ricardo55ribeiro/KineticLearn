from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def relative_error_against_prediction(true_values: np.ndarray, predicted_values: np.ndarray, epsilon: float = 1e-30) -> np.ndarray:
    denominator = predicted_values.copy()
    denominator[np.abs(denominator) < epsilon] = epsilon
    return np.abs((predicted_values - true_values) / denominator)


def plot_predicted_vs_true(
    targets_scaled: np.ndarray,
    outputs_scaled: np.ndarray,
    output_path: Path,
    regime_name: str,
) -> None:
    output_dim = targets_scaled.shape[1]
    plt.rcParams.update({"font.size": 14, "text.usetex": False})
    fig, axes = plt.subplots(1, output_dim, figsize=(5 * output_dim, 5), sharey=True)
    if output_dim == 1:
        axes = [axes]

    for output_idx in range(output_dim):
        ax = axes[output_idx]
        ax.scatter(targets_scaled[:, output_idx], outputs_scaled[:, output_idx], alpha=0.8, color=(0.0, 0.0, 0.9))
        ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), "--", color="black")
        ax.set_xlabel("True Values")
        if output_idx == 0:
            ax.set_ylabel("Predicted Values")
        ax.set_title(f"$k_{{{output_idx + 1}}}$")

        rel_err = relative_error_against_prediction(
            true_values=targets_scaled[:, output_idx],
            predicted_values=outputs_scaled[:, output_idx],
            epsilon=1e-9,
        )
        max_index = int(np.argmax(rel_err))
        ax.scatter(
            targets_scaled[max_index, output_idx],
            outputs_scaled[max_index, output_idx],
            color="gold",
            zorder=2,
        )
        text = "\n".join(
            (
                rf"$Mean\ \delta_{{rel}}={100.0 * rel_err.mean():.2f}\%$",
                rf"$Max\ \delta_{{rel}}={100.0 * rel_err.max():.2f}\%$",
                rf"$Regime: {regime_name}$",
            )
        )
        ax.text(0.58, 0.28, text, fontsize=11, transform=ax.transAxes, verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.5))
        if output_idx > 0:
            ax.tick_params(left=False)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def save_predictions_csv(
    output_path: Path,
    targets_scaled: np.ndarray,
    outputs_scaled_mean: np.ndarray,
    targets_unscaled: np.ndarray,
    outputs_unscaled_mean: np.ndarray,
    outputs_scaled_std: np.ndarray | None = None,
    outputs_unscaled_std: np.ndarray | None = None,
) -> None:
    n_samples, n_outputs = targets_scaled.shape
    data: dict[str, Any] = {"sample_id": np.arange(n_samples)}

    for output_idx in range(n_outputs):
        abs_err = np.abs(outputs_unscaled_mean[:, output_idx] - targets_unscaled[:, output_idx])
        sq_err = (outputs_unscaled_mean[:, output_idx] - targets_unscaled[:, output_idx]) ** 2
        rel_err = relative_error_against_prediction(
            true_values=targets_unscaled[:, output_idx],
            predicted_values=outputs_unscaled_mean[:, output_idx],
            epsilon=1e-30,
        )

        data[f"k{output_idx + 1}_true_scaled"] = targets_scaled[:, output_idx]
        data[f"k{output_idx + 1}_pred_scaled_mean"] = outputs_scaled_mean[:, output_idx]
        if outputs_scaled_std is not None:
            data[f"k{output_idx + 1}_pred_scaled_std"] = outputs_scaled_std[:, output_idx]

        data[f"k{output_idx + 1}_true_unscaled"] = targets_unscaled[:, output_idx]
        data[f"k{output_idx + 1}_pred_unscaled_mean"] = outputs_unscaled_mean[:, output_idx]
        if outputs_unscaled_std is not None:
            data[f"k{output_idx + 1}_pred_unscaled_std"] = outputs_unscaled_std[:, output_idx]

        data[f"k{output_idx + 1}_abs_err"] = abs_err
        data[f"k{output_idx + 1}_sq_err"] = sq_err
        data[f"k{output_idx + 1}_rel_err"] = rel_err

    pd.DataFrame(data).to_csv(output_path, index=False)


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
    parts: list[str] = ["Architecture comparison summary", ""]
    for row in rows:
        parts.append(f"Architecture: {row.get('architecture')}")
        for key, value in row.items():
            if key == "architecture":
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
