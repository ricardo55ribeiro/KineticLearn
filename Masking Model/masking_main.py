from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from masking_config import TrainingConfig, build_experiment_config, find_project_root
from masking_dataset import InverseKineticsDataset, build_feature_names
from masking_model import MaskingMLP
from masking_training import MaskingConfig, evaluate_observed_species_count, train_masked_model
from masking_utils import (
    arch_to_folder_name,
    make_results_root,
    plot_architecture_curves,
    plot_loss_curves,
    prepare_results_folders,
    save_global_summary_text,
    save_json,
    save_loss_history_csv,
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


def build_run_seed_bundle(seed: int) -> Dict[str, int]:
    seed = int(seed)
    return {
        "global": seed,
        "split": seed,
        "shuffle": seed + 100,
        "weights": seed,
        "training_masks": seed + 200,
        "validation_masks": seed + 300,
        "evaluation_base": seed + 400,
    }


def save_seed_run_metadata(
    seed_dir: Path,
    model: MaskingMLP,
    architecture: tuple[int, ...],
    seed: int,
    experiment_config,
    training_duration_s: float,
    history: dict[str, list[float]],
) -> None:
    metadata = {
        "architecture": list(architecture),
        "seed": int(seed),
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
    save_json(seed_dir / "model_info.json", metadata)


def save_seed_evaluation_outputs(
    seed_dir: Path,
    evaluation_rows: list[dict[str, Any]],
    sampled_subsets_by_count: dict[str, Any],
) -> None:
    save_summary_csv(seed_dir / "evaluation_by_observed_species.csv", evaluation_rows)
    save_global_summary_text(seed_dir / "evaluation_by_observed_species.txt", evaluation_rows)
    save_json(seed_dir / "evaluation_by_observed_species.json", {"rows": evaluation_rows})
    save_json(seed_dir / "sampled_subsets_by_observed_species.json", sampled_subsets_by_count)


def aggregate_metric_arrays(metric_arrays_by_seed: List[np.ndarray]) -> Dict[str, float]:
    if len(metric_arrays_by_seed) == 0:
        return {
            "mean": float("nan"),
            "between_seed_std": float("nan"),
            "within_subset_std": float("nan"),
            "total_std": float("nan"),
        }

    seed_means = np.asarray(
        [np.asarray(arr, dtype=float).mean() for arr in metric_arrays_by_seed],
        dtype=float,
    )
    within_vars = np.asarray(
        [
            np.asarray(arr, dtype=float).var(ddof=0) if len(arr) > 1 else 0.0
            for arr in metric_arrays_by_seed
        ],
        dtype=float,
    )

    between_var = float(seed_means.var(ddof=0)) if len(seed_means) > 1 else 0.0
    within_var = float(within_vars.mean()) if len(within_vars) > 0 else 0.0
    total_var = between_var + within_var

    return {
        "mean": float(seed_means.mean()),
        "between_seed_std": float(np.sqrt(between_var)),
        "within_subset_std": float(np.sqrt(within_var)),
        "total_std": float(np.sqrt(total_var)),
    }


def aggregate_architecture_seed_records(
    seed_records: List[dict[str, Any]],
    num_species: int,
    output_dim: int,
) -> List[dict[str, Any]]:
    if not seed_records:
        return []

    architecture_label = str(seed_records[0]["architecture"])
    counts = sorted({int(record["observed_species_count"]) for record in seed_records})
    all_seeds = sorted({int(record["seed"]) for record in seed_records})

    baseline_test_mse_by_seed: Dict[int, float] = {}
    baseline_test_mse_unscaled_by_seed: Dict[int, float] = {}

    for record in seed_records:
        if int(record["observed_species_count"]) == int(num_species):
            seed = int(record["seed"])
            baseline_test_mse_by_seed[seed] = float(np.asarray(record["per_repeat_metrics"]["test_mse"], dtype=float)[0])
            baseline_test_mse_unscaled_by_seed[seed] = float(
                np.asarray(record["per_repeat_metrics"]["test_mse_unscaled"], dtype=float)[0]
            )

    missing_baselines = [seed for seed in all_seeds if seed not in baseline_test_mse_by_seed]
    if missing_baselines:
        raise RuntimeError(
            f"Missing all-species baseline evaluation for seeds: {missing_baselines}"
        )

    metric_names = [
        "test_mse",
        "test_rmse",
        "test_mse_unscaled",
        "test_rmse_unscaled",
        "mean_rel_error_avg",
    ] + [f"mean_rel_error_k{i + 1}" for i in range(output_dim)]

    aggregated_rows: List[dict[str, Any]] = []

    for observed_species_count in counts:
        count_records = [
            record for record in seed_records
            if int(record["observed_species_count"]) == int(observed_species_count)
        ]
        count_records = sorted(count_records, key=lambda rec: int(rec["seed"]))

        row: Dict[str, Any] = {
            "architecture": architecture_label,
            "observed_species_count": int(observed_species_count),
            "n_seeds": int(len(count_records)),
            "repeats_per_seed": float(np.mean([int(record["repeats"]) for record in count_records])),
        }

        for metric_name in metric_names:
            metric_arrays = [
                np.asarray(record["per_repeat_metrics"][metric_name], dtype=float)
                for record in count_records
            ]
            stats = aggregate_metric_arrays(metric_arrays)
            row[metric_name] = stats["mean"]
            row[f"{metric_name}_between_seed_std"] = stats["between_seed_std"]
            row[f"{metric_name}_within_subset_std"] = stats["within_subset_std"]
            row[f"{metric_name}_total_std"] = stats["total_std"]

        relative_deterioration_scaled_arrays: List[np.ndarray] = []
        relative_deterioration_unscaled_arrays: List[np.ndarray] = []

        for record in count_records:
            seed = int(record["seed"])

            test_mse_values = np.asarray(record["per_repeat_metrics"]["test_mse"], dtype=float)
            baseline_test_mse = baseline_test_mse_by_seed[seed]

            test_mse_unscaled_values = np.asarray(record["per_repeat_metrics"]["test_mse_unscaled"], dtype=float)
            baseline_test_mse_unscaled = baseline_test_mse_unscaled_by_seed[seed]

            if int(observed_species_count) == int(num_species):
                det_scaled = np.zeros_like(test_mse_values)
                det_unscaled = np.zeros_like(test_mse_unscaled_values)
            else:
                det_scaled = 100.0 * (test_mse_values - baseline_test_mse) / baseline_test_mse
                det_unscaled = 100.0 * (test_mse_unscaled_values - baseline_test_mse_unscaled) / baseline_test_mse_unscaled

            relative_deterioration_scaled_arrays.append(det_scaled)
            relative_deterioration_unscaled_arrays.append(det_unscaled)

        for det_name, det_arrays in [
            ("relative_deterioration_test_mse_pct", relative_deterioration_scaled_arrays),
            ("relative_deterioration_test_mse_unscaled_pct", relative_deterioration_unscaled_arrays),
        ]:
            stats = aggregate_metric_arrays(det_arrays)
            row[det_name] = stats["mean"]
            row[f"{det_name}_between_seed_std"] = stats["between_seed_std"]
            row[f"{det_name}_within_subset_std"] = stats["within_subset_std"]
            row[f"{det_name}_total_std"] = stats["total_std"]

        aggregated_rows.append(row)

    return aggregated_rows


def create_analysis_outputs(
    results_root: Path,
    all_aggregated_rows: List[dict[str, Any]],
    architecture_order: List[str],
    output_dim: int,
) -> None:
    analysis_root = results_root / "Comparative_Analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    save_summary_csv(analysis_root / "aggregated_results.csv", all_aggregated_rows)
    save_global_summary_text(analysis_root / "aggregated_results.txt", all_aggregated_rows)

    plot_architecture_curves(
        aggregated_rows=all_aggregated_rows,
        output_path=analysis_root / "test_mse_vs_num_observed_species.pdf",
        y_col="test_mse",
        yerr_col="test_mse_total_std",
        title="Test MSE vs number of observed species",
        y_label="Test MSE",
        yscale="log",
        architecture_order=architecture_order,
    )

    plot_architecture_curves(
        aggregated_rows=all_aggregated_rows,
        output_path=analysis_root / "test_mse_unscaled_vs_num_observed_species.pdf",
        y_col="test_mse_unscaled",
        yerr_col="test_mse_unscaled_total_std",
        title="Test MSE (unscaled) vs number of observed species",
        y_label="Test MSE (unscaled)",
        yscale="log",
        architecture_order=architecture_order,
    )

    plot_architecture_curves(
        aggregated_rows=all_aggregated_rows,
        output_path=analysis_root / "mean_relative_error_avg_vs_num_observed_species.pdf",
        y_col="mean_rel_error_avg",
        yerr_col="mean_rel_error_avg_total_std",
        title="Mean relative error vs number of observed species",
        y_label="Mean relative error (average over k's)",
        yscale="log",
        architecture_order=architecture_order,
    )

    plot_architecture_curves(
        aggregated_rows=all_aggregated_rows,
        output_path=analysis_root / "relative_deterioration_test_mse_pct_vs_num_observed_species.pdf",
        y_col="relative_deterioration_test_mse_pct",
        yerr_col="relative_deterioration_test_mse_pct_total_std",
        title="Relative deterioration in test MSE vs number of observed species",
        y_label="Relative deterioration in test MSE (%)",
        yscale="symlog",
        architecture_order=architecture_order,
    )

    for output_idx in range(output_dim):
        plot_architecture_curves(
            aggregated_rows=all_aggregated_rows,
            output_path=analysis_root / f"mean_relative_error_k{output_idx + 1}_vs_num_observed_species.pdf",
            y_col=f"mean_rel_error_k{output_idx + 1}",
            yerr_col=f"mean_rel_error_k{output_idx + 1}_total_std",
            title=f"Mean relative error k{output_idx + 1} vs number of observed species",
            y_label=f"Mean relative error k{output_idx + 1}",
            yscale="log",
            architecture_order=architecture_order,
        )

    # Also save one filtered CSV/TXT per architecture for convenience.
    for architecture_label in architecture_order:
        arch_rows = [row for row in all_aggregated_rows if row["architecture"] == architecture_label]
        arch_analysis_dir = analysis_root / architecture_label
        arch_analysis_dir.mkdir(parents=True, exist_ok=True)

        save_summary_csv(arch_analysis_dir / "aggregated_results.csv", arch_rows)
        save_global_summary_text(arch_analysis_dir / "aggregated_results.txt", arch_rows)

        plot_architecture_curves(
            aggregated_rows=arch_rows,
            output_path=arch_analysis_dir / "test_mse_vs_num_observed_species.pdf",
            y_col="test_mse",
            yerr_col="test_mse_total_std",
            title=f"{architecture_label} — Test MSE vs number of observed species",
            y_label="Test MSE",
            yscale="log",
            architecture_order=[architecture_label],
        )

        plot_architecture_curves(
            aggregated_rows=arch_rows,
            output_path=arch_analysis_dir / "test_mse_unscaled_vs_num_observed_species.pdf",
            y_col="test_mse_unscaled",
            yerr_col="test_mse_unscaled_total_std",
            title=f"{architecture_label} — Test MSE (unscaled) vs number of observed species",
            y_label="Test MSE (unscaled)",
            yscale="log",
            architecture_order=[architecture_label],
        )

        plot_architecture_curves(
            aggregated_rows=arch_rows,
            output_path=arch_analysis_dir / "mean_relative_error_avg_vs_num_observed_species.pdf",
            y_col="mean_rel_error_avg",
            yerr_col="mean_rel_error_avg_total_std",
            title=f"{architecture_label} — Mean relative error vs number of observed species",
            y_label="Mean relative error (average over k's)",
            yscale="log",
            architecture_order=[architecture_label],
        )

        plot_architecture_curves(
            aggregated_rows=arch_rows,
            output_path=arch_analysis_dir / "relative_deterioration_test_mse_pct_vs_num_observed_species.pdf",
            y_col="relative_deterioration_test_mse_pct",
            yerr_col="relative_deterioration_test_mse_pct_total_std",
            title=f"{architecture_label} — Relative deterioration in test MSE vs number of observed species",
            y_label="Relative deterioration in test MSE (%)",
            yscale="symlog",
            architecture_order=[architecture_label],
        )

        for output_idx in range(output_dim):
            plot_architecture_curves(
                aggregated_rows=arch_rows,
                output_path=arch_analysis_dir / f"mean_relative_error_k{output_idx + 1}_vs_num_observed_species.pdf",
                y_col=f"mean_rel_error_k{output_idx + 1}",
                yerr_col=f"mean_rel_error_k{output_idx + 1}_total_std",
                title=f"{architecture_label} — Mean relative error k{output_idx + 1} vs number of observed species",
                y_label=f"Mean relative error k{output_idx + 1}",
                yscale="log",
                architecture_order=[architecture_label],
            )


def main() -> None:
    current_file = Path(__file__).resolve()
    project_root = find_project_root(current_file.parent)

    training_cfg = TrainingConfig()
    experiment_config = build_experiment_config(project_root=project_root, training_cfg=training_cfg)

    # Top-level determinism.
    set_global_seed(experiment_config.training.seed_global)

    dataset_train, dataset_test = build_datasets(experiment_config)
    x_test_unscaled, _ = dataset_test.get_unscaled_data()

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

    architectures = [tuple(int(x) for x in arch) for arch in experiment_config.training.architectures]
    architecture_labels = [arch_to_folder_name(architecture) for architecture in architectures]
    evaluation_counts = list(experiment_config.evaluation_species_counts)
    full_observed_species_count = int(experiment_config.dataset.num_species)

    total_runs = len(architectures) * len(experiment_config.training.seeds)
    all_seed_summary_rows: list[dict[str, Any]] = []
    all_aggregated_rows: list[dict[str, Any]] = []

    with tqdm(total=total_runs, desc="Masking full run", dynamic_ncols=True, position=0) as run_pbar:
        for architecture in architectures:
            architecture_label = arch_to_folder_name(architecture)
            arch_dir = arch_dirs[architecture]

            architecture_seed_records: list[dict[str, Any]] = []
            architecture_seed_summary_rows: list[dict[str, Any]] = []

            for seed in experiment_config.training.seeds:
                run_pbar.set_postfix(arch=architecture_label, seed=int(seed))

                seed_dir = arch_dir / f"seed_{int(seed):04d}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                seed_bundle = build_run_seed_bundle(int(seed))
                set_global_seed(seed_bundle["global"])
                torch.manual_seed(seed_bundle["weights"])

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
                        seed=seed_bundle["training_masks"],
                    ),
                    learning_rate=experiment_config.training.learning_rate,
                    batch_size=experiment_config.training.batch_size,
                    max_epochs=experiment_config.training.max_epochs,
                    patience=experiment_config.training.patience,
                    val_split=experiment_config.training.val_split,
                    device=experiment_config.training.device,
                    split_seed=seed_bundle["split"],
                    shuffle_seed=seed_bundle["shuffle"],
                    training_mask_seed=seed_bundle["training_masks"],
                    validation_mask_seed=seed_bundle["validation_masks"],
                )
                training_duration_s = time.time() - start_time

                model.load_state_dict(training_artifacts.model_state_dict)
                model.to(experiment_config.training.device)

                torch.save(model.state_dict(), seed_dir / "model.pth")
                save_seed_run_metadata(
                    seed_dir=seed_dir,
                    model=model,
                    architecture=architecture,
                    seed=int(seed),
                    experiment_config=experiment_config,
                    training_duration_s=training_duration_s,
                    history=training_artifacts.history,
                )
                save_loss_history_csv(seed_dir / "loss_history.csv", training_artifacts.history)
                plot_loss_curves(training_artifacts.history, seed_dir / "MaskingNetwork_loss_curves.pdf", log_scale=True)
                save_json(
                    seed_dir / "train_val_split.json",
                    {
                        "train_subset_indices": training_artifacts.train_subset_indices,
                        "val_subset_indices": training_artifacts.val_subset_indices,
                    },
                )

                x_test_scaled, y_test_scaled = dataset_test.get_data()

                seed_evaluation_rows: list[dict[str, Any]] = []
                sampled_subsets_by_count: dict[str, Any] = {}

                for observed_species_count in evaluation_counts:
                    repeats = (
                        1
                        if int(observed_species_count) == full_observed_species_count
                        else int(experiment_config.training.num_eval_repeats_per_count)
                    )
                    evaluation_seed = seed_bundle["evaluation_base"] + int(observed_species_count)

                    evaluation_artifact = evaluate_observed_species_count(
                        model=model,
                        x_scaled=x_test_scaled,
                        y_scaled=y_test_scaled,
                        inverse_transform_targets=dataset_test.inverse_transform_targets,
                        observed_species_count=int(observed_species_count),
                        species_names=experiment_config.dataset.species_names,
                        num_pressure_conditions=experiment_config.dataset.num_pressure_conditions,
                        repeats=repeats,
                        seed=evaluation_seed,
                        device=experiment_config.training.device,
                    )

                    seed_row = {
                        "architecture": architecture_label,
                        "seed": int(seed),
                        "training_time_s": float(training_duration_s),
                        "epochs_ran": len(training_artifacts.history["train_loss"]),
                        "best_val_loss": float(min(training_artifacts.history["val_loss"])),
                        **evaluation_artifact.metrics,
                    }
                    seed_evaluation_rows.append(seed_row)
                    architecture_seed_summary_rows.append(seed_row)
                    all_seed_summary_rows.append(seed_row)

                    sampled_subsets_by_count[str(observed_species_count)] = evaluation_artifact.sampled_species_subsets

                    architecture_seed_records.append(
                        {
                            "architecture": architecture_label,
                            "seed": int(seed),
                            "observed_species_count": int(observed_species_count),
                            "repeats": int(repeats),
                            "per_repeat_metrics": evaluation_artifact.per_repeat_metrics,
                        }
                    )

                save_seed_evaluation_outputs(
                    seed_dir=seed_dir,
                    evaluation_rows=seed_evaluation_rows,
                    sampled_subsets_by_count=sampled_subsets_by_count,
                )

                run_pbar.update(1)

            save_summary_csv(arch_dir / "seed_level_evaluation_summary.csv", architecture_seed_summary_rows)
            save_global_summary_text(arch_dir / "seed_level_evaluation_summary.txt", architecture_seed_summary_rows)

            architecture_aggregated_rows = aggregate_architecture_seed_records(
                seed_records=architecture_seed_records,
                num_species=full_observed_species_count,
                output_dim=experiment_config.output_dim,
            )
            save_summary_csv(arch_dir / "aggregated_by_observed_species.csv", architecture_aggregated_rows)
            save_global_summary_text(arch_dir / "aggregated_by_observed_species.txt", architecture_aggregated_rows)

            all_aggregated_rows.extend(architecture_aggregated_rows)

    save_summary_csv(results_root / "seed_level_evaluation_summary.csv", all_seed_summary_rows)
    save_global_summary_text(results_root / "seed_level_evaluation_summary.txt", all_seed_summary_rows)

    save_summary_csv(results_root / "summary.csv", all_aggregated_rows)
    save_global_summary_text(results_root / "summary.txt", all_aggregated_rows)

    create_analysis_outputs(
        results_root=results_root,
        all_aggregated_rows=all_aggregated_rows,
        architecture_order=architecture_labels,
        output_dim=experiment_config.output_dim,
    )

    print(f"\nAll results saved under: {results_root}")


if __name__ == "__main__":
    main()
