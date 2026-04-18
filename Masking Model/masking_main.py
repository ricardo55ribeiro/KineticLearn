from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from masking_config import TrainingConfig, build_experiment_config, find_project_root
from masking_dataset import InverseKineticsDataset, build_feature_names
from masking_model import MaskingMLP
from masking_training import MaskingConfig, evaluate_regime, train_masked_model
from masking_utils import (
    arch_to_folder_name,
    make_results_root,
    metrics_to_text,
    plot_loss_curves,
    plot_predicted_vs_true,
    prepare_results_folders,
    save_global_summary_text,
    save_json,
    save_loss_history_csv,
    save_predictions_csv,
    save_summary_csv,
    save_test_inputs_csv,
    set_global_seed,
)


def build_datasets(experiment_config):
    dataset_train = InverseKineticsDataset(
        src_file=experiment_config.dataset.main_dataset,
        num_species=experiment_config.dataset.num_species,
        num_pressure_conditions=experiment_config.dataset.num_pressure_conditions,
        k_columns=experiment_config.dataset.k_columns,
        multiply_targets_by=experiment_config.training.multiply_targets_by,
    )
    dataset_test = InverseKineticsDataset(
        src_file=experiment_config.dataset.main_dataset_test,
        num_species=experiment_config.dataset.num_species,
        num_pressure_conditions=experiment_config.dataset.num_pressure_conditions,
        k_columns=experiment_config.dataset.k_columns,
        multiply_targets_by=experiment_config.training.multiply_targets_by,
        input_scalers=dataset_train.scalers.input_scalers,
        output_scaler=dataset_train.scalers.output_scaler,
    )
    return dataset_train, dataset_test


def save_architecture_metadata(arch_dir: Path, model: MaskingMLP, experiment_config, training_duration_s: float, history: dict[str, list[float]]) -> None:
    metadata = {
        "architecture": list(model.hidden_sizes),
        "activation": model.activation,
        "input_dim": experiment_config.model_input_dim,
        "output_dim": experiment_config.output_dim,
        "num_parameters": model.count_parameters(),
        "epochs_ran": len(history["train_loss"]),
        "best_epoch": int(np.argmin(history["val_loss"]) + 1),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "best_val_loss": float(min(history["val_loss"])),
        "training_time_s": float(training_duration_s),
    }
    save_json(arch_dir / "model_info.json", metadata)


def save_evaluation_outputs(
    arch_dir: Path,
    regime_artifact,
    y_scaled_np: np.ndarray,
    y_unscaled_np: np.ndarray,
) -> None:
    regime_name = regime_artifact.regime_name

    predictions_csv_path = arch_dir / f"predictions__{regime_name}.csv"
    save_predictions_csv(
        output_path=predictions_csv_path,
        targets_scaled=y_scaled_np,
        outputs_scaled_mean=regime_artifact.predictions_scaled_mean,
        targets_unscaled=y_unscaled_np,
        outputs_unscaled_mean=regime_artifact.predictions_unscaled_mean,
        outputs_scaled_std=regime_artifact.predictions_scaled_std if regime_artifact.repeats > 1 else None,
        outputs_unscaled_std=regime_artifact.predictions_unscaled_std if regime_artifact.repeats > 1 else None,
    )

    plot_predicted_vs_true(
        targets_scaled=y_scaled_np,
        outputs_scaled_mean=regime_artifact.predictions_scaled_mean,
        output_path=arch_dir / f"MaskingNetwork__{regime_name}.pdf",
        regime_name=regime_name,
    )

    subset_info = {
        "regime_name": regime_name,
        "repeats": regime_artifact.repeats,
        "observed_species_counts": regime_artifact.observed_species_counts,
        "sampled_species_subsets": regime_artifact.sampled_species_subsets,
    }
    save_json(arch_dir / f"sampled_subsets__{regime_name}.json", subset_info)


def main() -> None:
    current_file = Path(__file__).resolve()
    project_root = find_project_root(current_file.parent)

    training_cfg = TrainingConfig()
    experiment_config = build_experiment_config(project_root=project_root, training_cfg=training_cfg)

    set_global_seed(experiment_config.training.seed_global)

    dataset_train, dataset_test = build_datasets(experiment_config)
    x_test_scaled, y_test_scaled = dataset_test.get_data()
    x_test_unscaled, y_test_unscaled = dataset_test.get_unscaled_data()
    y_test_scaled_np = y_test_scaled.numpy()
    y_test_unscaled_np = y_test_unscaled.numpy()

    feature_names = build_feature_names(
        species_names=experiment_config.dataset.species_names,
        num_pressure_conditions=experiment_config.dataset.num_pressure_conditions,
    )

    results_root = make_results_root(
        base_root=project_root / experiment_config.training.results_root_name,
        scheme=experiment_config.dataset.scheme,
        experiment_name=experiment_config.experiment_name,
        add_timestamp=True,
    )
    arch_dirs = prepare_results_folders(experiment_config.training.architectures, results_root)

    save_json(results_root / "experiment_info.json", experiment_config.to_dict())
    save_test_inputs_csv(results_root / "test_inputs.csv", x_test_unscaled.numpy(), feature_names)

    primary_regime_name = experiment_config.evaluation_regimes[-1].name
    summary_rows: list[dict[str, Any]] = []

    for architecture in experiment_config.training.architectures:
        architecture = tuple(int(x) for x in architecture)
        arch_dir = arch_dirs[architecture]
        print(f"\n--- Training architecture: {architecture} ---")

        torch.manual_seed(experiment_config.training.seed_weights)
        model = MaskingMLP(
            input_dim=experiment_config.model_input_dim,
            output_dim=experiment_config.output_dim,
            hidden_sizes=architecture,
            activation=experiment_config.training.activation,
        )

        start_time = time.time()
        training_artifacts = train_masked_model(
            model=model,
            dataset_train=dataset_train,
            masking_config=MaskingConfig(
                num_species=experiment_config.dataset.num_species,
                num_pressure_conditions=experiment_config.dataset.num_pressure_conditions,
                min_observed_species=experiment_config.training.min_observed_species_train,
                max_observed_species=experiment_config.dataset.num_species,
                seed=experiment_config.training.seed_training_masks,
            ),
            learning_rate=experiment_config.training.learning_rate,
            batch_size=experiment_config.training.batch_size,
            max_epochs=experiment_config.training.max_epochs,
            patience=experiment_config.training.patience,
            val_split=experiment_config.training.val_split,
            device=experiment_config.training.device,
            split_seed=experiment_config.training.seed_split,
            shuffle_seed=experiment_config.training.seed_shuffle,
            training_mask_seed=experiment_config.training.seed_training_masks,
            validation_mask_seed=experiment_config.training.seed_validation_masks,
        )
        training_duration_s = time.time() - start_time

        model.load_state_dict(training_artifacts.model_state_dict)
        model.to(experiment_config.training.device)

        torch.save(model.state_dict(), arch_dir / "model.pth")
        save_architecture_metadata(
            arch_dir=arch_dir,
            model=model,
            experiment_config=experiment_config,
            training_duration_s=training_duration_s,
            history=training_artifacts.history,
        )
        save_loss_history_csv(arch_dir / "loss_history.csv", training_artifacts.history)
        plot_loss_curves(training_artifacts.history, arch_dir / "MaskingNetwork_loss_curves.pdf", log_scale=True)

        split_info = {
            "train_subset_indices": training_artifacts.train_subset_indices,
            "val_subset_indices": training_artifacts.val_subset_indices,
        }
        save_json(arch_dir / "train_val_split.json", split_info)

        regime_metrics_rows: list[dict[str, Any]] = []
        regime_metrics_json: dict[str, Any] = {}

        for regime_index, regime in enumerate(experiment_config.evaluation_regimes):
            regime_seed = experiment_config.training.seed_evaluation_masks + regime_index
            regime_artifact = evaluate_regime(
                model=model,
                x_scaled=x_test_scaled,
                y_scaled=y_test_scaled,
                inverse_transform_targets=dataset_test.inverse_transform_targets,
                regime_name=regime.name,
                regime_mode=regime.mode,
                species_names=experiment_config.dataset.species_names,
                num_pressure_conditions=experiment_config.dataset.num_pressure_conditions,
                min_observed_species_train=experiment_config.training.min_observed_species_train,
                repeats=regime.repeats,
                seed=regime_seed,
                device=experiment_config.training.device,
                exact_species_count=regime.exact_species_count,
            )

            save_evaluation_outputs(
                arch_dir=arch_dir,
                regime_artifact=regime_artifact,
                y_scaled_np=y_test_scaled_np,
                y_unscaled_np=y_test_unscaled_np,
            )

            regime_metrics_rows.append(regime_artifact.metrics)
            regime_metrics_json[regime.name] = regime_artifact.metrics

        save_json(arch_dir / "evaluation_metrics.json", regime_metrics_json)
        save_summary_csv(arch_dir / "evaluation_metrics.csv", regime_metrics_rows)

        primary_metrics = regime_metrics_json[primary_regime_name]
        arch_summary_row = {
            "architecture": arch_to_folder_name(architecture),
            "num_parameters": model.count_parameters(),
            "training_time_s": training_duration_s,
            "epochs_ran": len(training_artifacts.history["train_loss"]),
            "best_val_loss": float(min(training_artifacts.history["val_loss"])),
            "primary_regime": primary_regime_name,
            **primary_metrics,
        }
        summary_rows.append(arch_summary_row)

        save_json(arch_dir / "primary_metrics.json", arch_summary_row)
        (arch_dir / "primary_metrics.txt").write_text(metrics_to_text(arch_summary_row), encoding="utf-8")

        print(
            f"Architecture {architecture} finished. "
            f"Primary regime RMSE (unscaled, mean over repeats): "
            f"{primary_metrics['rmse_unscaled_mean_over_repeats']:.6e}"
        )

    save_summary_csv(results_root / "summary.csv", summary_rows)
    save_global_summary_text(results_root / "summary.txt", summary_rows)
    print(f"\nAll results saved under: {results_root}")


if __name__ == "__main__":
    main()
