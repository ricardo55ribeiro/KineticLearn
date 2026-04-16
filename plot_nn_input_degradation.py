import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ZOOM_MIN = 3

SCHEME = "O2_novib"
SCHEME_ROOT = Path("Results_NN") / SCHEME
ANALYSIS_DIR = SCHEME_ROOT / "Comparative_Analysis"

BASELINE_PREFIX = "11__"

IGNORED_DIR_NAMES = {
    "Comparative_Analysis",
    "Comparative_Plots",
    "Plots",
}

# Exact architecture folders to create inside Comparative_Analysis
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


def get_latest_run_dir(experiment_dir: Path):
    run_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda p: p.name)[-1]


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


def load_all_results():
    rows = []

    if not SCHEME_ROOT.exists():
        raise FileNotFoundError(f"Scheme root not found: {SCHEME_ROOT}")

    for experiment_dir in SCHEME_ROOT.iterdir():
        if not experiment_dir.is_dir():
            continue

        if experiment_dir.name in IGNORED_DIR_NAMES:
            continue

        run_dir = get_latest_run_dir(experiment_dir)
        if run_dir is None:
            continue

        summary_csv = run_dir / "summary.csv"
        if not summary_csv.exists():
            print(f"Skipping {experiment_dir.name}: summary.csv not found")
            continue

        df = pd.read_csv(summary_csv)
        df["experiment_folder"] = experiment_dir.name
        df["run_timestamp"] = run_dir.name
        rows.append(df)

    if not rows:
        raise RuntimeError("No summary.csv files were found under the scheme root.")

    all_df = pd.concat(rows, ignore_index=True)

    numeric_cols = [
        "input_size",
        "num_species_kept",
        "num_pressure_conditions",
        "test_mse",
        "test_rmse",
        "training_time_s",
        "mean_rel_error_k1",
        "mean_rel_error_k2",
        "mean_rel_error_k3",
        "max_rel_error_k1",
        "max_rel_error_k2",
        "max_rel_error_k3",
    ]
    for col in numeric_cols:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    if "hidden_size" not in all_df.columns:
        raise RuntimeError("summary.csv does not contain a 'hidden_size' column.")

    all_df["hidden_size_tuple"] = all_df["hidden_size"].apply(parse_hidden_size)
    all_df["hidden_size_label"] = all_df["hidden_size_tuple"].apply(hidden_size_to_label)
    all_df["arch_sort_key"] = all_df["hidden_size_tuple"].apply(hidden_size_sort_key)

    if "num_species_kept" not in all_df.columns or all_df["num_species_kept"].isna().all():
        if "input_size" in all_df.columns:
            n_pressures = 2
            if "num_pressure_conditions" in all_df.columns:
                vals = all_df["num_pressure_conditions"].dropna().unique()
                if len(vals) == 1:
                    n_pressures = int(vals[0])
            all_df["num_species_kept"] = all_df["input_size"] / n_pressures
        else:
            raise RuntimeError("Could not determine the number of kept species.")

    all_df["num_species_kept"] = pd.to_numeric(all_df["num_species_kept"], errors="coerce")
    all_df["num_species_kept"] = all_df["num_species_kept"].round().astype("Int64")

    rel_cols = [c for c in all_df.columns if c.startswith("mean_rel_error_k")]
    if rel_cols:
        all_df["mean_rel_error_avg"] = all_df[rel_cols].mean(axis=1)

    all_df = all_df.replace([np.inf, -np.inf], np.nan)

    # Keep only the three architectures requested
    all_df = all_df[all_df["hidden_size_label"].isin(ARCHITECTURES)].copy()

    if all_df.empty:
        raise RuntimeError("No rows found for the requested architectures.")

    return all_df


def compute_relative_deterioration(df):
    baseline_mask = df["experiment_folder"].astype(str).str.startswith(BASELINE_PREFIX)
    baseline_df = df[baseline_mask].copy()

    if baseline_df.empty:
        raise RuntimeError(
            f"No baseline experiment found. Expected at least one experiment folder starting with '{BASELINE_PREFIX}'."
        )

    baseline_df = baseline_df[
        ["hidden_size_label", "num_species_kept", "test_mse"]
    ].rename(
        columns={
            "num_species_kept": "baseline_num_species",
            "test_mse": "baseline_test_mse",
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

    merged["mse_ratio_vs_baseline"] = merged["test_mse"] / merged["baseline_test_mse"]

    merged["relative_deterioration_test_mse_pct"] = (
        100.0 * (merged["test_mse"] - merged["baseline_test_mse"])
        / merged["baseline_test_mse"]
    )

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
        "hidden_size_label",
        "num_species_kept",
        "test_mse",
        "baseline_num_species",
        "baseline_test_mse",
        "mse_ratio_vs_baseline",
        "relative_deterioration_test_mse_pct",
    ]

    extra_cols = [
        "mean_rel_error_avg",
        "mean_rel_error_k1",
        "mean_rel_error_k2",
        "mean_rel_error_k3",
        "max_rel_error_k1",
        "max_rel_error_k2",
        "max_rel_error_k3",
        "training_time_s",
    ]
    for col in extra_cols:
        if col in df_arch.columns:
            cols.append(col)

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
    lines.append("")

    baseline_candidates = df_arch[df_arch["experiment_folder"].astype(str).str.startswith(BASELINE_PREFIX)]
    if not baseline_candidates.empty:
        baseline_row = baseline_candidates.iloc[0]
        lines.append(f"Baseline experiment: {baseline_row['experiment_folder']}")
        lines.append(f"Baseline number of species: {int(baseline_row['baseline_num_species'])}")
        lines.append(f"Baseline test MSE: {baseline_row['baseline_test_mse']:.6e}")
        lines.append("")

    for _, row in df_arch.sort_values("num_species_kept").iterrows():
        ratio = row["mse_ratio_vs_baseline"]
        signed_det = row["relative_deterioration_test_mse_pct"]

        ratio_str = "nan" if pd.isna(ratio) else f"{ratio:.4f}"
        signed_det_str = "nan" if pd.isna(signed_det) else f"{signed_det:+.2f}%"

        lines.append(
            f"Species: {int(row['num_species_kept']):>2d} | "
            f"Experiment: {row['experiment_folder']} | "
            f"Test MSE: {row['test_mse']:.6e} | "
            f"MSE ratio: {ratio_str} | "
            f"Relative deterioration: {signed_det_str}"
        )

    with open(arch_dir / "relative_deterioration_report.txt", "w") as f:
        f.write("\n".join(lines))


def make_single_arch_plot(
    df_arch,
    arch_label,
    y_col,
    y_label,
    filename,
    title=None,
    yscale=None,
    min_species=None,
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

    plt.figure(figsize=(8, 6))
    plt.plot(
        plot_df["num_species_kept"],
        plot_df[y_col],
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
    # Main performance plot
    make_single_arch_plot(
        df_arch=df_arch,
        arch_label=arch_label,
        y_col="test_mse",
        y_label="Test MSE",
        filename=arch_dir / "test_mse_vs_num_species.pdf",
        title=f"{arch_label} — Test MSE vs number of species (zoomed: {ZOOM_MIN} to 11 species)",
        yscale="log",
    )

    # Zoomed performance plot
    make_single_arch_plot(
        df_arch=df_arch,
        arch_label=arch_label,
        y_col="test_mse",
        y_label="Test MSE",
        filename=arch_dir / "test_mse_vs_num_species_zoom.pdf",
        title=f"{arch_label} — Test MSE vs number of species (zoomed: 4 to 11 species)",
        yscale="log",
        min_species=ZOOM_MIN,
    )

    # Relative deterioration plot
    make_single_arch_plot(
        df_arch=df_arch,
        arch_label=arch_label,
        y_col="relative_deterioration_test_mse_pct",
        y_label="Relative deterioration in test MSE (%)",
        filename=arch_dir / "relative_deterioration_vs_num_species.pdf",
        title=f"{arch_label} — Relative deterioration vs number of species (zoomed: {ZOOM_MIN} to 11 species)",
        yscale="symlog",
    )

    # Zoomed relative deterioration plot
    make_single_arch_plot(
        df_arch=df_arch,
        arch_label=arch_label,
        y_col="relative_deterioration_test_mse_pct",
        y_label="Relative deterioration in test MSE (%)",
        filename=arch_dir / "relative_deterioration_vs_num_species_zoom.pdf",
        title=f"{arch_label} — Relative deterioration vs number of species (zoomed: 4 to 11 species)",
        yscale=None,
        min_species=ZOOM_MIN,
    )

    # Average mean relative error
    if "mean_rel_error_avg" in df_arch.columns:
        make_single_arch_plot(
            df_arch=df_arch,
            arch_label=arch_label,
            y_col="mean_rel_error_avg",
            y_label="Mean relative error (average over k's)",
            filename=arch_dir / "mean_relative_error_avg_vs_num_species.pdf",
            title=f"{arch_label} — Mean relative error vs number of species (zoomed: {ZOOM_MIN} to 11 species)",
            yscale="log",
        )

        make_single_arch_plot(
            df_arch=df_arch,
            arch_label=arch_label,
            y_col="mean_rel_error_avg",
            y_label="Mean relative error (average over k's)",
            filename=arch_dir / "mean_relative_error_avg_vs_num_species_zoom.pdf",
            title=f"{arch_label} — Mean relative error vs number of species (zoomed: 4 to 11 species)",
            yscale="log",
            min_species=ZOOM_MIN,
        )

    # Per-k relative-error plots
    for k in [1, 2, 3]:
        col = f"mean_rel_error_k{k}"
        if col in df_arch.columns:
            make_single_arch_plot(
                df_arch=df_arch,
                arch_label=arch_label,
                y_col=col,
                y_label=f"Mean relative error k{k}",
                filename=arch_dir / f"mean_relative_error_k{k}_vs_num_species.pdf",
                title=f"{arch_label} — Mean relative error k{k} vs number of species (zoomed: {ZOOM_MIN} to 11 species)",
                yscale="log",
            )

            make_single_arch_plot(
                df_arch=df_arch,
                arch_label=arch_label,
                y_col=col,
                y_label=f"Mean relative error k{k}",
                filename=arch_dir / f"mean_relative_error_k{k}_vs_num_species_zoom.pdf",
                title=f"{arch_label} — Mean relative error k{k} vs number of species (zoomed: 4 to 11 species)",
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
