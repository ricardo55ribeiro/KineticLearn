import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------------------------------------------
# Import shared cache/training implementation

try:
    import InverseProblem_NoiseRobustness as shared
except ImportError as exc:
    raise ImportError(
        "Could not import InverseProblem_NoiseRobustness.py.\n"
        "Place Train_SavedWeights.py in the same project-root folder as "
        "InverseProblem_NoiseRobustness.py and run it from there."
    ) from exc


REQUIRED_SHARED_ATTRIBUTES = [
    "SAVED_WEIGHTS_ROOT",
    "saved_scheme_root",
    "saved_model_dir",
    "species_config_to_name",
    "arch_to_folder_name",
    "validate_all_species_configs",
    "load_datasets_with_saved_scalers",
    "get_or_train_model",
    "save_json",
    "dictionary",
]

missing = [name for name in REQUIRED_SHARED_ATTRIBUTES if not hasattr(shared, name)]
if missing:
    raise AttributeError(
        "The imported InverseProblem_NoiseRobustness.py does not contain the shared "
        "saved_weights helpers required by this trainer.\n"
        f"Missing: {missing}\n"
        "Use the rewritten shared-cache version of InverseProblem_NoiseRobustness.py."
    )


# --------------------------------------------------------------------------------------
# User setup

SCHEME = "O2_novib"

# O2(X) /  O2(a)  /  O2(b)  / O2(Hz) / O2+(X) / O(3P)
# O(1D) / O+(gnd) / O-(gnd) /  O3(X) / O3(exc)
SPECIES_CONFIGS = [
    ["O2(a)", "O2(b)", "O+(gnd)"],
    ["O2(a)", "O2(b)", "O2+(X)"],
    ["O2(X)", "O2(a)", "O2(b)", "O2(Hz)", "O2+(X)", "O(3P)", "O(1D)", "O+(gnd)", "O-(gnd)", "O3(X)"],
    ["O2(X)", "O2(a)", "O2(b)", "O2(Hz)", "O2+(X)", "O(3P)", "O(1D)", "O+(gnd)", "O-(gnd)"],
    ["O2(X)", "O2(a)", "O2(b)", "O2(Hz)", "O2+(X)", "O(3P)", "O(1D)", "O+(gnd)"],
    ["O2(X)", "O2(a)", "O2(b)", "O2(Hz)", "O2+(X)", "O(3P)", "O(1D)"],
    ["O2(X)", "O2(a)", "O2(b)", "O2(Hz)", "O2+(X)", "O(3P)"],
    ["O2(X)", "O2(a)", "O2(b)", "O2(Hz)", "O2+(X)"],
]

ARCHITECTURES = [
    (30, 30),
    (50, 50),
    (30, 30, 30),
]

# 20 seeds: 32, 33, ..., 51
SEEDS = list(range(32, 52))

ACTIVATION = "tanh"
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
MAX_EPOCHS = 5000
PATIENCE = 100
VAL_SPLIT = 0.1
VERBOSE_EPOCH_LOSSES = False

# False:
#     Reuse compatible saved weights if they already exist.
#     Train only missing or incompatible models.
#
# True:
#     Delete each matching seed/architecture cache folder before training,
#     forcing fresh training.
FORCE_RETRAIN = False

# Summary files saved directly under saved_weights/<SCHEME>/.
SAVE_SUMMARY_FILES = True
SUMMARY_FILENAME = "pretrain_summary.csv"
AGGREGATE_FILENAME = "pretrain_aggregate_summary.csv"
INFO_FILENAME = "pretrain_info.json"


# --------------------------------------------------------------------------------------
# Helpers

def deduplicate_species_configs(species_configs):
    """Preserve order while removing duplicate species configurations."""
    unique = []
    seen = set()

    for config in species_configs:
        key = tuple(config)
        if key in seen:
            continue
        seen.add(key)
        unique.append(list(config))

    return unique


def stringify_list(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value))
    return value


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def maybe_force_delete_cached_model(scheme, kept_species, seed, hidden_size):
    if not FORCE_RETRAIN:
        return

    model_dir = shared.saved_model_dir(scheme, kept_species, seed, hidden_size)
    if model_dir.exists():
        shutil.rmtree(model_dir)


def build_summary_row(
    scheme,
    kept_species,
    hidden_size,
    seed,
    dataset_train,
    dataset_test,
    info,
    training_record,
):
    x_train, y_train = dataset_train.get_data()
    x_test, y_test = dataset_test.get_data()

    model_dir = shared.saved_model_dir(scheme, kept_species, seed, hidden_size)
    model_path = model_dir / "model.pth"
    info_path = model_dir / "model_info.json"
    loss_history_path = model_dir / "loss_history.csv"
    clean_metrics_path = model_dir / "clean_metrics.json"

    return {
        "scheme": scheme,
        "species_config_name": shared.species_config_to_name(kept_species),
        "kept_species": list(kept_species),
        "num_species_kept": int(len(kept_species)),
        "hidden_size": list(hidden_size),
        "hidden_size_str": shared.arch_to_folder_name(hidden_size),
        "seed": int(seed),
        "input_size": int(x_train.shape[1]),
        "output_size": int(y_train.shape[1]),
        "x_train_shape": list(x_train.shape),
        "y_train_shape": list(y_train.shape),
        "x_test_shape": list(x_test.shape),
        "y_test_shape": list(y_test.shape),
        "activation": ACTIVATION,
        "learning_rate": float(LEARNING_RATE),
        "batch_size": int(BATCH_SIZE),
        "max_epochs": int(MAX_EPOCHS),
        "patience": int(PATIENCE),
        "val_split": float(VAL_SPLIT),
        "reused_saved_weights": bool(training_record.get("reused_saved_weights", False)),
        "saved_weights_path": str(model_path),
        "model_info_path": str(info_path),
        "loss_history_path": str(loss_history_path),
        "clean_metrics_path": str(clean_metrics_path),
        "model_file_exists": bool(model_path.exists()),
        "model_info_exists": bool(info_path.exists()),
        "loss_history_exists": bool(loss_history_path.exists()),
        "clean_metrics_exists": bool(clean_metrics_path.exists()),
        "epochs_ran": int(training_record.get("epochs_ran", info.get("epochs_ran", 0))),
        "best_epoch": int(training_record.get("best_epoch", info.get("best_epoch", 0))),
        "final_train_loss": safe_float(training_record.get("final_train_loss", info.get("final_train_loss", np.nan))),
        "final_val_loss": safe_float(training_record.get("final_val_loss", info.get("final_val_loss", np.nan))),
        "best_val_loss": safe_float(training_record.get("best_val_loss", info.get("best_val_loss", np.nan))),
        "current_run_training_time_s": safe_float(training_record.get("training_time_s", 0.0), 0.0),
        "cached_training_time_s": safe_float(training_record.get("cached_training_time_s", info.get("training_time_s", 0.0)), 0.0),
        "test_mse_scaled": safe_float(info.get("test_mse_scaled", np.nan)),
        "test_rmse_scaled": safe_float(info.get("test_rmse_scaled", np.nan)),
    }


def save_summary_tables(scheme, rows):
    root = shared.saved_scheme_root(scheme)
    root.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    if df.empty:
        df.to_csv(root / SUMMARY_FILENAME, index=False)
        return

    summary_df = df.copy()
    summary_df["kept_species"] = summary_df["kept_species"].apply(stringify_list)
    summary_df["hidden_size"] = summary_df["hidden_size"].apply(stringify_list)
    summary_df["x_train_shape"] = summary_df["x_train_shape"].apply(stringify_list)
    summary_df["y_train_shape"] = summary_df["y_train_shape"].apply(stringify_list)
    summary_df["x_test_shape"] = summary_df["x_test_shape"].apply(stringify_list)
    summary_df["y_test_shape"] = summary_df["y_test_shape"].apply(stringify_list)
    summary_df.to_csv(root / SUMMARY_FILENAME, index=False)

    df["reused_saved_weights_int"] = df["reused_saved_weights"].astype(int)
    aggregate = (
        df.groupby(
            [
                "scheme",
                "species_config_name",
                "num_species_kept",
                "hidden_size_str",
            ],
            as_index=False,
        )
        .agg(
            num_seeds=("seed", "nunique"),
            num_models=("seed", "count"),
            num_reused_saved_weights=("reused_saved_weights_int", "sum"),
            fraction_reused_saved_weights=("reused_saved_weights_int", "mean"),
            mean_test_mse_scaled=("test_mse_scaled", "mean"),
            std_test_mse_scaled=("test_mse_scaled", "std"),
            min_test_mse_scaled=("test_mse_scaled", "min"),
            max_test_mse_scaled=("test_mse_scaled", "max"),
            mean_best_val_loss=("best_val_loss", "mean"),
            std_best_val_loss=("best_val_loss", "std"),
            mean_epochs_ran=("epochs_ran", "mean"),
            mean_current_run_training_time_s=("current_run_training_time_s", "mean"),
            total_current_run_training_time_s=("current_run_training_time_s", "sum"),
            mean_cached_training_time_s=("cached_training_time_s", "mean"),
        )
    )

    aggregate.rename(columns={"hidden_size_str": "hidden_size"}, inplace=True)
    aggregate.to_csv(root / AGGREGATE_FILENAME, index=False)


def save_run_info(scheme, species_configs):
    root = shared.saved_scheme_root(scheme)
    root.mkdir(parents=True, exist_ok=True)

    info = {
        "script": "Train_SavedWeights.py",
        "purpose": (
            "Pretrain/reuse neural-network models into the shared saved_weights cache "
            "without running noise robustness or full-result workflows."
        ),
        "scheme": scheme,
        "saved_weights_root": str(shared.SAVED_WEIGHTS_ROOT),
        "scheme_saved_weights_root": str(root),
        "species_configs": species_configs,
        "architectures": [list(a) for a in ARCHITECTURES],
        "seeds": SEEDS,
        "num_species_configs": int(len(species_configs)),
        "num_architectures": int(len(ARCHITECTURES)),
        "num_seeds": int(len(SEEDS)),
        "num_planned_models": int(len(species_configs) * len(ARCHITECTURES) * len(SEEDS)),
        "activation": ACTIVATION,
        "learning_rate": float(LEARNING_RATE),
        "batch_size": int(BATCH_SIZE),
        "max_epochs": int(MAX_EPOCHS),
        "patience": int(PATIENCE),
        "val_split": float(VAL_SPLIT),
        "force_retrain": bool(FORCE_RETRAIN),
        "cache_policy": (
            "If FORCE_RETRAIN is False, load compatible saved_weights models; "
            "train and save only missing or incompatible models. "
            "If FORCE_RETRAIN is True, delete each matching model directory first."
        ),
        "summary_files": {
            "summary_csv": str(root / SUMMARY_FILENAME),
            "aggregate_csv": str(root / AGGREGATE_FILENAME),
        },
    }

    shared.save_json(root / INFO_FILENAME, info)


# --------------------------------------------------------------------------------------
# Main workflow

def main():
    species_configs = deduplicate_species_configs(SPECIES_CONFIGS)
    shared.validate_all_species_configs(species_configs)

    shared.SAVED_WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    shared.saved_scheme_root(SCHEME).mkdir(parents=True, exist_ok=True)

    save_run_info(SCHEME, species_configs)

    total_models = len(species_configs) * len(ARCHITECTURES) * len(SEEDS)
    rows = []

    print(f"Shared saved-weights root: {shared.SAVED_WEIGHTS_ROOT}")
    print(f"Scheme cache folder: {shared.saved_scheme_root(SCHEME)}")
    print(f"Species configurations: {len(species_configs)}")
    print(f"Architectures: {len(ARCHITECTURES)}")
    print(f"Seeds: {len(SEEDS)} ({SEEDS[0]} to {SEEDS[-1]})")
    print(f"Total planned models: {total_models}")
    print(f"FORCE_RETRAIN: {FORCE_RETRAIN}")

    with tqdm(total=total_models, desc="Pretraining saved_weights") as pbar:
        for kept_species in species_configs:
            dataset_train, dataset_test = shared.load_datasets_with_saved_scalers(SCHEME, kept_species)
            species_name = shared.species_config_to_name(kept_species)

            for hidden_size in ARCHITECTURES:
                arch_name = shared.arch_to_folder_name(hidden_size)

                for seed in SEEDS:
                    maybe_force_delete_cached_model(SCHEME, kept_species, seed, hidden_size)

                    model, info, loss_history, training_record = shared.get_or_train_model(
                        scheme=SCHEME,
                        kept_species=kept_species,
                        hidden_size=hidden_size,
                        seed=seed,
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

                    row = build_summary_row(
                        scheme=SCHEME,
                        kept_species=kept_species,
                        hidden_size=hidden_size,
                        seed=seed,
                        dataset_train=dataset_train,
                        dataset_test=dataset_test,
                        info=info,
                        training_record=training_record,
                    )
                    rows.append(row)

                    status = "reused" if row["reused_saved_weights"] else "trained"
                    mse = row["test_mse_scaled"]
                    mse_text = "nan" if np.isnan(mse) else f"{mse:.3e}"

                    pbar.set_postfix(
                        species=species_name,
                        arch=arch_name,
                        seed=seed,
                        status=status,
                        mse=mse_text,
                    )
                    pbar.update(1)

                    # Free the model reference before the next run.
                    del model

    if SAVE_SUMMARY_FILES:
        save_summary_tables(SCHEME, rows)

    num_reused = sum(1 for row in rows if row["reused_saved_weights"])
    num_trained = len(rows) - num_reused

    print("")
    print("Finished saved_weights pretraining.")
    print(f"Models checked: {len(rows)}")
    print(f"Reused compatible models: {num_reused}")
    print(f"Newly trained models: {num_trained}")
    print(f"Saved weights folder: {shared.saved_scheme_root(SCHEME)}")

    if SAVE_SUMMARY_FILES:
        print(f"Summary CSV: {shared.saved_scheme_root(SCHEME) / SUMMARY_FILENAME}")
        print(f"Aggregate CSV: {shared.saved_scheme_root(SCHEME) / AGGREGATE_FILENAME}")


if __name__ == "__main__":
    main()
