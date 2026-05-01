import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SPECIES_MIN = 2
ZOOM_MIN = 3

SCHEME = "O2_novib"
SCHEME_ROOT = Path("Results_NN") / SCHEME
FULLRUN_PARENT = SCHEME_ROOT / "FullRun_2to11species"
ANALYSIS_DIR = FULLRUN_PARENT / "Comparative_Analysis"

BASELINE_PREFIX = "11__"

IGNORED_DIR_NAMES = {
    "Comparative_Analysis",
    "Comparative_Plots",
    "Plots",
}

ARCHITECTURES = [
    "30, 30",
    "30, 30, 30",
    "50, 50",
]

ARCH_COLORS = {
    "30, 30": "blue",
    "30, 30, 30": "green",
    "50, 50": "red",
}

TIMESTAMP_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
EXPERIMENT_REGEX = re.compile(r"^\d+__")


def parse_hidden_size(value):
    if isinstance(value, tuple):
        return tuple(int(x) for x in value)

    if isinstance(value, list):
        return tuple(int(x) for x in value)

    if pd.isna(value):
        return tuple()

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return tuple(int(x) for x in parsed)
    except Exception:
        pass

    text = text.strip("[]()")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def hidden_size_to_label(hidden_size_tuple):
    return ", ".join(map(str, hidden_size_tuple))


def hidden_size_sort_key(hidden_size_tuple):
    if not hidden_size_tuple:
        return (999, 999, ())
    return (len(hidden_size_tuple), hidden_size_tuple[0], hidden_size_tuple)


def get_color_for_architecture(label):
    return ARCH_COLORS.get(label, "black")


def find_timestamp_ancestor(path: Path):
    for ancestor in [path, *path.parents]:
        if TIMESTAMP_REGEX.match(ancestor.name):
            return ancestor.name
    return ""


def timestamp_sort_key(text: str):
    return text if text else ""


def is_valid_experiment_dir(path: Path):
    return path.is_dir() and EXPERIMENT_REGEX.match(path.name) and path.name not in IGNORED_DIR_NAMES


def summarize_summary_df(df, experiment_folder, run_timestamp):
    if "hidden_size" not in df.columns:
        raise RuntimeError(f"summary.csv in {experiment_folder} does not contain 'hidden_size'.")

    numeric_candidates = [
        "input_size",
        "num_species_kept",
        "num_pressure_conditions",
        "test_mse",
        "test_rmse",
        "test_mse_unscaled",
        "test_rmse_unscaled",
        "training_time_s",
        "epochs_ran",
        "best_val_loss",
        "mean_rel_error_k1",
        "mean_rel_error_k2",
        "mean_rel_error_k3",
        "max_rel_error_k1",
        "max_rel_error_k2",
        "max_rel_error_k3",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["hidden_size_tuple"] = df["hidden_size"].apply(parse_hidden_size)
    df["hidden_size_label"] = df["hidden_size_tuple"].apply(hidden_size_to_label)

    if "num_species_kept" not in df.columns or df["num_species_kept"].isna().all():
        if "input_size" in df.columns:
            n_pressures = 2
            if "num_pressure_conditions" in df.columns:
                vals = df["num_pressure_conditions"].dropna().unique()
                if len(vals) == 1:
                    n_pressures = int(vals[0])
            df["num_species_kept"] = df["input_size"] / n_pressures
        else:
            raise RuntimeError(f"Could not determine num_species_kept for {experiment_folder}.")

    df["num_species_kept"] = pd.to_numeric(df["num_species_kept"], errors="coerce").round().astype("Int64")

    group_cols = [
        "experiment_folder",
        "run_timestamp",
        "hidden_size_label",
        "hidden_size_tuple",
        "num_species_kept",
    ]

    metric_cols = [
        "test_mse",
        "test_rmse",
        "test_mse_unscaled",
        "test_rmse_unscaled",
        "training_time_s",
        "epochs_ran",
        "best_val_loss",
        "mean_rel_error_k1",
        "mean_rel_error_k2",
        "mean_rel_error_k3",
        "max_rel_error_k1",
        "max_rel_error_k2",
        "max_rel_error_k3",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    df["experiment_folder"] = experiment_folder
    df["run_timestamp"] = run_timestamp

    agg_dict = {col: ["mean", "std"] for col in metric_cols}
    grouped = df.groupby(group_cols, as_index=False).agg(agg_dict)
    grouped.columns = [
        col if isinstance(col, str) else "_".join([c for c in col if c])
        for col in grouped.columns.to_flat_index()
    ]

    rename_map = {}
    for col in metric_cols:
        rename_map[f"{col}_mean"] = col
        rename_map[f"{col}_std"] = f"{col}_std"
    grouped = grouped.rename(columns=rename_map)

    grouped["source_mode"] = "summary_aggregated" if len(df) > len(grouped) else "summary_single"
    return grouped


def summarize_seed_aggregate_df(df, experiment_folder, run_timestamp):
    hidden_size_col = None
    if "hidden_size" in df.columns:
        hidden_size_col = "hidden_size"
    elif "hidden_size_str" in df.columns:
        hidden_size_col = "hidden_size_str"
    else:
        raise RuntimeError(
            f"seed_aggregate_summary.csv does not contain 'hidden_size' or 'hidden_size_str'. "
            f"Columns found: {list(df.columns)}"
        )

    numeric_cols = [
        col for col in df.columns
        if col not in {"scheme", "experiment_name", "hidden_size", "hidden_size_str"}
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["hidden_size_tuple"] = df[hidden_size_col].apply(parse_hidden_size)
    df["hidden_size_label"] = df["hidden_size_tuple"].apply(hidden_size_to_label)
    df["num_species_kept"] = pd.to_numeric(df["num_species_kept"], errors="coerce").round().astype("Int64")

    out = pd.DataFrame({
        "experiment_folder": experiment_folder,
        "run_timestamp": run_timestamp,
        "hidden_size_label": df["hidden_size_label"],
        "hidden_size_tuple": df["hidden_size_tuple"],
        "num_species_kept": df["num_species_kept"],
        "source_mode": "seed_aggregate",
    })

    mean_std_pairs = [
        "test_mse",
        "test_rmse",
        "test_mse_unscaled",
        "test_rmse_unscaled",
        "training_time_s",
        "epochs_ran",
        "best_val_loss",
        "mean_rel_error_k1",
        "mean_rel_error_k2",
        "mean_rel_error_k3",
        "max_rel_error_k1",
        "max_rel_error_k2",
        "max_rel_error_k3",
    ]

    for metric in mean_std_pairs:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in df.columns:
            out[metric] = df[mean_col]
        if std_col in df.columns:
            out[f"{metric}_std"] = df[std_col]

    return out


def get_latest_fullrun_dir():
    if not FULLRUN_PARENT.exists():
        raise FileNotFoundError(f"FullRun folder not found: {FULLRUN_PARENT}")

    run_dirs = [
        d for d in FULLRUN_PARENT.iterdir()
        if d.is_dir() and TIMESTAMP_REGEX.match(d.name)
    ]

    if not run_dirs:
        raise RuntimeError(f"No timestamp run folders found inside: {FULLRUN_PARENT}")

    return sorted(run_dirs, key=lambda p: p.name)[-1]

def collect_latest_experiment_tables():
    if not SCHEME_ROOT.exists():
        raise FileNotFoundError(f"Scheme root not found: {SCHEME_ROOT}")

    candidates = {}

    search_root = get_latest_fullrun_dir()

    for csv_path in search_root.rglob("*.csv"):
        if csv_path.name not in {"summary.csv", "seed_aggregate_summary.csv"}:
            continue

        experiment_dir = csv_path.parent
        if not is_valid_experiment_dir(experiment_dir):
            continue

        experiment_name = experiment_dir.name
        run_timestamp = find_timestamp_ancestor(experiment_dir)
        source_priority = 1 if csv_path.name == "seed_aggregate_summary.csv" else 0

        info = {
            "path": csv_path,
            "run_timestamp": run_timestamp,
            "source_priority": source_priority,
            "mtime": csv_path.stat().st_mtime,
        }

        prev = candidates.get(experiment_name)
        if prev is None:
            candidates[experiment_name] = info
            continue

        prev_key = (timestamp_sort_key(prev["run_timestamp"]), prev["source_priority"], prev["mtime"])
        curr_key = (timestamp_sort_key(info["run_timestamp"]), info["source_priority"], info["mtime"])
        if curr_key > prev_key:
            candidates[experiment_name] = info

    if not candidates:
        raise RuntimeError("No suitable summary.csv or seed_aggregate_summary.csv files were found.")

    tables = []
    for experiment_name, info in sorted(candidates.items()):
        csv_path = info["path"]
        run_timestamp = info["run_timestamp"]
        df = pd.read_csv(csv_path)

        if csv_path.name == "seed_aggregate_summary.csv":
            tables.append(summarize_seed_aggregate_df(df, experiment_name, run_timestamp))
        else:
            tables.append(summarize_summary_df(df, experiment_name, run_timestamp))

    return tables


def load_all_results():
    rows = collect_latest_experiment_tables()
    all_df = pd.concat(rows, ignore_index=True)

    all_df["hidden_size_tuple"] = all_df["hidden_size_tuple"].apply(parse_hidden_size)
    all_df["hidden_size_label"] = all_df["hidden_size_tuple"].apply(hidden_size_to_label)
    all_df["arch_sort_key"] = all_df["hidden_size_tuple"].apply(hidden_size_sort_key)

    rel_cols = [c for c in all_df.columns if re.fullmatch(r"mean_rel_error_k\d+", c)]
    if rel_cols:
        all_df["mean_rel_error_avg"] = all_df[rel_cols].mean(axis=1)

    rel_std_cols = [c for c in all_df.columns if re.fullmatch(r"mean_rel_error_k\d+_std", c)]
    if rel_std_cols:
        sq = sum(np.square(all_df[c].fillna(0.0)) for c in rel_std_cols)
        counts = sum(all_df[c].notna().astype(int) for c in rel_std_cols)
        with np.errstate(invalid="ignore", divide="ignore"):
            all_df["mean_rel_error_avg_std"] = np.sqrt(sq) / counts.replace(0, np.nan)

    keep_cols = [c for c in all_df.columns if c.startswith("test_") or c.startswith("mean_rel_error") or c.startswith("max_rel_error")]
    for col in keep_cols + ["training_time_s", "epochs_ran", "best_val_loss"]:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    all_df = all_df.replace([np.inf, -np.inf], np.nan)
    all_df = all_df[all_df["hidden_size_label"].isin(ARCHITECTURES)].copy()

    if all_df.empty:
        raise RuntimeError("No rows found for the requested architectures.")

    return all_df


def propagate_ratio_std(numerator, numerator_std, denominator, denominator_std):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator_std = pd.to_numeric(numerator_std, errors="coerce")
    denominator_std = pd.to_numeric(denominator_std, errors="coerce")

    ratio = numerator / denominator

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_num = numerator_std / numerator
        rel_den = denominator_std / denominator
        ratio_std = ratio * np.sqrt(np.square(rel_num) + np.square(rel_den))

    ratio_std = ratio_std.where(
        numerator_std.notna() & denominator_std.notna() & (numerator > 0) & (denominator > 0),
        np.nan,
    )
    return ratio, ratio_std


def compute_relative_deterioration(df):
    baseline_mask = df["experiment_folder"].astype(str).str.startswith(BASELINE_PREFIX)
    baseline_df = df[baseline_mask].copy()

    if baseline_df.empty:
        raise RuntimeError(
            f"No baseline experiment found. Expected at least one experiment folder starting with '{BASELINE_PREFIX}'."
        )

    baseline_df = baseline_df[
        [
            "hidden_size_label",
            "num_species_kept",
            "test_mse",
            *(["test_mse_std"] if "test_mse_std" in baseline_df.columns else []),
        ]
    ].rename(
        columns={
            "num_species_kept": "baseline_num_species",
            "test_mse": "baseline_test_mse",
            "test_mse_std": "baseline_test_mse_std",
        }
    )

    merged = df.merge(baseline_df, on="hidden_size_label", how="left")

    if merged["baseline_test_mse"].isna().any():
        missing_archs = (
            merged.loc[merged["baseline_test_mse"].isna(), "hidden_size_label"]
            .dropna()
            .unique()
            .tolist()
        )
        raise RuntimeError(f"Missing baseline for some architectures: {missing_archs}")

    if "test_mse_std" not in merged.columns:
        merged["test_mse_std"] = np.nan
    if "baseline_test_mse_std" not in merged.columns:
        merged["baseline_test_mse_std"] = np.nan

    ratio, ratio_std = propagate_ratio_std(
        merged["test_mse"],
        merged["test_mse_std"],
        merged["baseline_test_mse"],
        merged["baseline_test_mse_std"],
    )

    merged["mse_ratio_vs_baseline"] = ratio
    merged["mse_ratio_vs_baseline_std"] = ratio_std
    merged["relative_deterioration_test_mse_pct"] = 100.0 * (ratio - 1.0)
    merged["relative_deterioration_test_mse_pct_std"] = 100.0 * ratio_std

    baseline_self_mask = merged["experiment_folder"].astype(str).str.startswith(BASELINE_PREFIX)
    merged.loc[baseline_self_mask, "mse_ratio_vs_baseline"] = 1.0
    merged.loc[baseline_self_mask, "mse_ratio_vs_baseline_std"] = 0.0
    merged.loc[baseline_self_mask, "relative_deterioration_test_mse_pct"] = 0.0
    merged.loc[baseline_self_mask, "relative_deterioration_test_mse_pct_std"] = 0.0

    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged


def prepare_architecture_dirs():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    arch_dirs = {}
    for arch in ARCHITECTURES:
        arch_dir = ANALYSIS_DIR / arch
        arch_dir.mkdir(parents=True, exist_ok=True)
        arch_dirs[arch] = arch_dir

    return arch_dirs


def save_aggregated_table(df):
    out_csv = ANALYSIS_DIR / "aggregated_results.csv"
    df.sort_values(["hidden_size_label", "num_species_kept"]).to_csv(out_csv, index=False)


def save_architecture_report_csv(arch_dir, df_arch):
    cols = [
        "experiment_folder",
        "run_timestamp",
        "source_mode",
        "hidden_size_label",
        "num_species_kept",
        "test_mse",
        "test_mse_std",
        "baseline_num_species",
        "baseline_test_mse",
        "baseline_test_mse_std",
        "mse_ratio_vs_baseline",
        "mse_ratio_vs_baseline_std",
        "relative_deterioration_test_mse_pct",
        "relative_deterioration_test_mse_pct_std",
    ]

    extra_cols = [
        "mean_rel_error_avg",
        "mean_rel_error_avg_std",
        "mean_rel_error_k1",
        "mean_rel_error_k1_std",
        "mean_rel_error_k2",
        "mean_rel_error_k2_std",
        "mean_rel_error_k3",
        "mean_rel_error_k3_std",
        "max_rel_error_k1",
        "max_rel_error_k2",
        "max_rel_error_k3",
        "training_time_s",
        "training_time_s_std",
    ]
    cols = [c for c in cols + extra_cols if c in df_arch.columns]

    out_df = df_arch[cols].sort_values("num_species_kept").copy()
    out_df.to_csv(arch_dir / "relative_deterioration_report.csv", index=False)


def save_architecture_report_txt(arch_dir, arch_label, df_arch):
    lines = []
    lines.append(f"Architecture report: {arch_label}")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  MSE ratio = MSE_current / MSE_baseline")
    lines.append("  Relative deterioration (%) = 100 * (MSE_current - MSE_baseline) / MSE_baseline")
    lines.append(f"  Baseline = experiment folder starting with '{BASELINE_PREFIX}' (11 kept species)")
    lines.append("  Error bars are shown only when a standard deviation is available.")
    lines.append("")

    baseline_candidates = df_arch[df_arch["experiment_folder"].astype(str).str.startswith(BASELINE_PREFIX)]
    if not baseline_candidates.empty:
        baseline_row = baseline_candidates.iloc[0]
        baseline_std_str = ""
        if "baseline_test_mse_std" in baseline_row and pd.notna(baseline_row["baseline_test_mse_std"]):
            baseline_std_str = f" ± {baseline_row['baseline_test_mse_std']:.6e}"
        lines.append(f"Baseline experiment: {baseline_row['experiment_folder']}")
        lines.append(f"Baseline number of species: {int(baseline_row['baseline_num_species'])}")
        lines.append(f"Baseline test MSE: {baseline_row['baseline_test_mse']:.6e}{baseline_std_str}")
        lines.append("")

    for _, row in df_arch.sort_values("num_species_kept").iterrows():
        ratio = row.get("mse_ratio_vs_baseline", np.nan)
        ratio_std = row.get("mse_ratio_vs_baseline_std", np.nan)
        signed_det = row.get("relative_deterioration_test_mse_pct", np.nan)
        signed_det_std = row.get("relative_deterioration_test_mse_pct_std", np.nan)
        mse_std = row.get("test_mse_std", np.nan)

        ratio_str = "nan" if pd.isna(ratio) else f"{ratio:.4f}"
        if pd.notna(ratio_std):
            ratio_str += f" ± {ratio_std:.4f}"

        signed_det_str = "nan" if pd.isna(signed_det) else f"{signed_det:+.2f}%"
        if pd.notna(signed_det_std):
            signed_det_str += f" ± {signed_det_std:.2f}%"

        mse_str = f"{row['test_mse']:.6e}"
        if pd.notna(mse_std):
            mse_str += f" ± {mse_std:.6e}"

        lines.append(
            f"Species: {int(row['num_species_kept']):>2d} | "
            f"Experiment: {row['experiment_folder']} | "
            f"Source: {row.get('source_mode', 'unknown')} | "
            f"Test MSE: {mse_str} | "
            f"MSE ratio: {ratio_str} | "
            f"Relative deterioration: {signed_det_str}"
        )

    with open(arch_dir / "relative_deterioration_report.txt", "w") as f:
        f.write("\n".join(lines))


def sanitize_yerr_for_log(y, yerr):
    if yerr is None:
        return None
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    yerr = np.where(np.isnan(yerr), 0.0, yerr)
    yerr = np.where(yerr < 0, 0.0, yerr)
    return np.minimum(yerr, np.maximum(0.0, 0.999 * y))


def make_single_arch_plot(
    df_arch,
    arch_label,
    y_col,
    y_label,
    filename,
    title=None,
    yscale=None,
    min_species=None,
    yerr_col=None,
):
    if y_col not in df_arch.columns:
        return

    plot_df = df_arch.copy()
    if min_species is not None:
        plot_df = plot_df[plot_df["num_species_kept"] >= min_species].copy()

    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["num_species_kept", y_col]).sort_values("num_species_kept")

    if plot_df.empty:
        return

    color = get_color_for_architecture(arch_label)

    x = plot_df["num_species_kept"].to_numpy(dtype=float)
    y = plot_df[y_col].to_numpy(dtype=float)

    use_errorbars = False
    yerr = None
    if yerr_col is not None and yerr_col in plot_df.columns:
        yerr_series = pd.to_numeric(plot_df[yerr_col], errors="coerce")
        if yerr_series.notna().any():
            yerr = yerr_series.to_numpy(dtype=float)
            if yscale == "log":
                yerr = sanitize_yerr_for_log(y, yerr)
            else:
                yerr = np.where(np.isnan(yerr), 0.0, yerr)
                yerr = np.where(yerr < 0, 0.0, yerr)
            use_errorbars = np.any(yerr > 0)

    plt.figure(figsize=(8, 6))

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
        )
    else:
        plt.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            color=color,
        )

    plt.xlabel("Number of Species")
    plt.ylabel(y_label)

    if title:
        plt.title(title)

    if yscale == "log":
        vals = plot_df[y_col].dropna()
        if not vals.empty and (vals > 0).all():
            plt.yscale("log")
    elif yscale == "symlog":
        plt.yscale("symlog", linthresh=1)

    xticks = sorted(plot_df["num_species_kept"].dropna().unique())
    plt.xticks(xticks)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def make_architecture_plots(arch_dir, arch_label, df_arch):
    make_single_arch_plot(
        df_arch=df_arch,
        arch_label=arch_label,
        y_col="test_mse",
        yerr_col="test_mse_std",
        y_label="Test MSE",
        filename=arch_dir / "test_mse_vs_num_species.pdf",
        title=f"{arch_label} — Test MSE vs number of species ({SPECIES_MIN} to 11 species)",
        yscale="log",
    )

    make_single_arch_plot(
        df_arch=df_arch,
        arch_label=arch_label,
        y_col="test_mse",
        yerr_col="test_mse_std",
        y_label="Test MSE",
        filename=arch_dir / "test_mse_vs_num_species_zoom.pdf",
        title=f"{arch_label} — Test MSE vs number of species ({ZOOM_MIN} to 11 species)",
        yscale="log",
        min_species=ZOOM_MIN,
    )

    make_single_arch_plot(
        df_arch=df_arch,
        arch_label=arch_label,
        y_col="relative_deterioration_test_mse_pct",
        y_label="Relative deterioration in test MSE (%)",
        filename=arch_dir / "relative_deterioration_vs_num_species.pdf",
        title=f"{arch_label} — Relative deterioration vs number of species ({SPECIES_MIN} to 11 species)",
        yscale="symlog",
    )

    make_single_arch_plot(
        df_arch=df_arch,
        arch_label=arch_label,
        y_col="relative_deterioration_test_mse_pct",
        y_label="Relative deterioration in test MSE (%)",
        filename=arch_dir / "relative_deterioration_vs_num_species_zoom.pdf",
        title=f"{arch_label} — Relative deterioration vs number of species ({ZOOM_MIN} to 11 species)",
        yscale=None,
        min_species=ZOOM_MIN,
    )

    if "mean_rel_error_avg" in df_arch.columns:
        make_single_arch_plot(
            df_arch=df_arch,
            arch_label=arch_label,
            y_col="mean_rel_error_avg",
            yerr_col="mean_rel_error_avg_std",
            y_label="Mean relative error (average over k's)",
            filename=arch_dir / "mean_relative_error_avg_vs_num_species.pdf",
            title=f"{arch_label} — Mean relative error vs number of species ({SPECIES_MIN} to 11 species)",
            yscale="log",
        )

        make_single_arch_plot(
            df_arch=df_arch,
            arch_label=arch_label,
            y_col="mean_rel_error_avg",
            yerr_col="mean_rel_error_avg_std",
            y_label="Mean relative error (average over k's)",
            filename=arch_dir / "mean_relative_error_avg_vs_num_species_zoom.pdf",
            title=f"{arch_label} — Mean relative error vs number of species ({ZOOM_MIN} to 11 species)",
            yscale="log",
            min_species=ZOOM_MIN,
        )

    for k in [1, 2, 3]:
        col = f"mean_rel_error_k{k}"
        std_col = f"mean_rel_error_k{k}_std"
        if col in df_arch.columns:
            make_single_arch_plot(
                df_arch=df_arch,
                arch_label=arch_label,
                y_col=col,
                yerr_col=std_col,
                y_label=f"Mean relative error k{k}",
                filename=arch_dir / f"mean_relative_error_k{k}_vs_num_species.pdf",
                title=f"{arch_label} — Mean relative error k{k} vs number of species ({SPECIES_MIN} to 11 species)",
                yscale="log",
            )

            make_single_arch_plot(
                df_arch=df_arch,
                arch_label=arch_label,
                y_col=col,
                yerr_col=std_col,
                y_label=f"Mean relative error k{k}",
                filename=arch_dir / f"mean_relative_error_k{k}_vs_num_species_zoom.pdf",
                title=f"{arch_label} — Mean relative error k{k} vs number of species ({ZOOM_MIN} to 11 species)",
                yscale="log",
                min_species=ZOOM_MIN,
            )


def main():
    df = load_all_results()
    df = compute_relative_deterioration(df)

    arch_dirs = prepare_architecture_dirs()
    save_aggregated_table(df)

    for arch_label in ARCHITECTURES:
        df_arch = df[df["hidden_size_label"] == arch_label].copy()

        if df_arch.empty:
            print(f"Skipping architecture {arch_label}: no rows found.")
            continue

        arch_dir = arch_dirs[arch_label]

        save_architecture_report_csv(arch_dir, df_arch)
        save_architecture_report_txt(arch_dir, arch_label, df_arch)
        make_architecture_plots(arch_dir, arch_label, df_arch)

    print(f"Analysis files saved to: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
