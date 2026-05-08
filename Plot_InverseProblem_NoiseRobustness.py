from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# User setup

SCHEME = "O2_novib"
EXPERIMENT_NAME = "InverseProblem_GaussianNoiseRobustness"
BASE_RESULTS_DIR = Path("Results_NN")

# Set to None to automatically use the latest timestamped noise-results folder.
# If your result CSVs are directly inside Noise_MSE_Results, leave this as None.
# Example manual value: NOISE_RESULTS_TIMESTAMP = "2026-05-04_15-30-00"
NOISE_RESULTS_TIMESTAMP = None

MAIN_METRIC = "test_mse_scaled"
MAIN_METRIC_LABEL = "Scaled MSE"
USE_LOG_Y = True
SAVE_PNG = True
SAVE_PDF = True
DPI = 300

# Requested high-contrast colours.
# If there are more than three lines, colours cycle and markers change.
PLOT_COLORS = ["green", "red", "blue"]
PLOT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]

MAX_LEGEND_COLUMNS = 2


# --------------------------------------------------------------------------------------
# Helpers


def safe_path_token(text):
    return (
        str(text)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
        .replace(" ", "")
        .replace(",", "_")
    )


def results_parent_root():
    return BASE_RESULTS_DIR / SCHEME / EXPERIMENT_NAME / "Noise_MSE_Results"


def has_noise_results(folder):
    folder = Path(folder)
    return (
        (folder / "fullrun_noise_aggregate_summary.csv").exists()
        or (folder / "fullrun_noise_results.csv").exists()
    )


def get_results_root():
    parent = results_parent_root()

    if not parent.exists():
        raise FileNotFoundError(
            f"Noise results folder does not exist:\n{parent}\n"
            "Run InverseProblem_NoiseRobustness.py first."
        )

    if NOISE_RESULTS_TIMESTAMP is not None:
        root = parent / NOISE_RESULTS_TIMESTAMP
        if not has_noise_results(root):
            raise FileNotFoundError(
                f"Could not find fullrun_noise_aggregate_summary.csv or "
                f"fullrun_noise_results.csv in:\n{root}"
            )
        return root

    # Support both layouts:
    #   Noise_MSE_Results/fullrun_noise_aggregate_summary.csv
    #   Noise_MSE_Results/<timestamp>/fullrun_noise_aggregate_summary.csv
    if has_noise_results(parent):
        return parent

    candidate_roots = [p for p in parent.iterdir() if p.is_dir() and has_noise_results(p)]
    if not candidate_roots:
        raise FileNotFoundError(
            f"Could not find fullrun_noise_aggregate_summary.csv or "
            f"fullrun_noise_results.csv in:\n{parent}\n"
            "Also checked timestamped subfolders."
        )

    return sorted(candidate_roots, key=lambda p: p.name)[-1]


def flatten_aggregate_if_needed(df):
    """Return an aggregate table with '<metric>_mean' and '<metric>_std' columns."""
    if f"{MAIN_METRIC}_mean" in df.columns:
        return df

    # Fallback if the input table is the raw results file.
    if MAIN_METRIC in df.columns:
        group_cols = [
            "scheme",
            "experiment_name",
            "species_config_name",
            "kept_species",
            "num_species_kept",
            "input_size",
            "output_size",
            "hidden_size",
            "noise_std",
            "noise_percent",
            "noise_label",
        ]
        missing_group_cols = [col for col in group_cols if col not in df.columns]
        if missing_group_cols:
            raise ValueError(
                "Raw results table is missing required grouping columns: "
                + ", ".join(missing_group_cols)
            )

        metric_cols = [col for col in df.columns if col.endswith("_scaled")]
        agg_dict = {col: ["mean", "std", "min", "max"] for col in metric_cols}
        agg = df.groupby(group_cols, as_index=False).agg(agg_dict)
        agg.columns = [
            col if isinstance(col, str) else "_".join([c for c in col if c])
            for col in agg.columns.to_flat_index()
        ]
        return agg

    raise ValueError(
        f"Could not find either {MAIN_METRIC!r} or {MAIN_METRIC + '_mean'!r} in the table."
    )


def load_aggregate_table(results_root):
    results_root = Path(results_root)
    agg_path = results_root / "fullrun_noise_aggregate_summary.csv"
    raw_path = results_root / "fullrun_noise_results.csv"

    if agg_path.exists():
        df = pd.read_csv(agg_path)
    elif raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        raise FileNotFoundError(
            f"Could not find aggregate or raw noise results in:\n{results_root}"
        )

    df = flatten_aggregate_if_needed(df)

    required_cols = [
        "species_config_name",
        "kept_species",
        "num_species_kept",
        "hidden_size",
        "noise_percent",
        f"{MAIN_METRIC}_mean",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns: " + ", ".join(missing_cols))

    df = df.sort_values(
        ["num_species_kept", "species_config_name", "hidden_size", "noise_percent"]
    ).reset_index(drop=True)
    return df


def metric_columns(metric):
    return (
        f"{metric}_mean",
        f"{metric}_std",
        f"{metric}_min",
        f"{metric}_max",
    )


def get_metric_values(df, metric):
    mean_col, std_col, _, _ = metric_columns(metric)
    if mean_col not in df.columns:
        raise ValueError(f"Missing required column: {mean_col}")

    y = df[mean_col].to_numpy(dtype=float)

    if std_col in df.columns:
        yerr = df[std_col].fillna(0.0).to_numpy(dtype=float)
    else:
        yerr = np.zeros_like(y)

    return y, yerr


def short_species_label(row_or_df):
    if isinstance(row_or_df, pd.DataFrame):
        species = str(row_or_df["kept_species"].iloc[0])
    else:
        species = str(row_or_df["kept_species"])

    # The saved CSV stores this as a comma-separated string already. This cleanup also
    # protects against accidental list-string formatting such as "['O2(X)', 'O2(a)']".
    species = (
        species.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )
    species = ", ".join(part.strip() for part in species.split(",") if part.strip())
    return species


def ordered_unique(series):
    return list(dict.fromkeys(series.tolist()))


def style_for(index):
    return {
        "color": PLOT_COLORS[index % len(PLOT_COLORS)],
        "marker": PLOT_MARKERS[index % len(PLOT_MARKERS)],
    }


def save_current_figure(output_base):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_PDF:
        plt.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    if SAVE_PNG:
        plt.savefig(output_base.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    plt.close()


def format_axes(ax):
    ax.set_xlabel("Gaussian noise level on input densities (%)", fontsize=13)
    ax.set_ylabel(MAIN_METRIC_LABEL, fontsize=13)
    if USE_LOG_Y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)


def set_two_line_title(ax, first_line, second_line):
    ax.set_title(f"{first_line}\n{second_line}", fontsize=13)


# --------------------------------------------------------------------------------------
# Plot families


def plot_by_species(df, plots_root):
    """Create per-species-combination folders.

    Each folder contains:
        - all_architectures: combined architecture comparison
        - one individual plot for each architecture
    """
    output_root = Path(plots_root) / "by_species"
    output_root.mkdir(parents=True, exist_ok=True)

    all_architectures = ordered_unique(df["hidden_size"])
    architecture_style = {arch: style_for(i) for i, arch in enumerate(all_architectures)}

    for species_config_name, species_df in df.groupby("species_config_name", sort=False):
        species_label = short_species_label(species_df)
        species_dir = output_root / safe_path_token(species_config_name)
        species_dir.mkdir(parents=True, exist_ok=True)

        # Combined plot: one line per architecture.
        fig, ax = plt.subplots(figsize=(9.5, 6.5))

        for hidden_size, arch_df in species_df.groupby("hidden_size", sort=False):
            arch_df = arch_df.sort_values("noise_percent")
            x = arch_df["noise_percent"].to_numpy(dtype=float)
            y, yerr = get_metric_values(arch_df, MAIN_METRIC)
            style = architecture_style[hidden_size]

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=style["marker"],
                color=style["color"],
                linewidth=1.8,
                capsize=4,
                label=f"{hidden_size}",
            )

        set_two_line_title(
            ax,
            "Inverse noise robustness by architecture",
            f"Species: {species_label}",
        )
        ax.set_xticks(sorted(species_df["noise_percent"].unique()))
        format_axes(ax)
        ax.legend(title="Architecture", fontsize=10, title_fontsize=10, frameon=False)
        fig.tight_layout()

        save_current_figure(species_dir / "all_architectures")

        # Individual plots: one architecture per plot.
        for hidden_size, arch_df in species_df.groupby("hidden_size", sort=False):
            arch_df = arch_df.sort_values("noise_percent")
            x = arch_df["noise_percent"].to_numpy(dtype=float)
            y, yerr = get_metric_values(arch_df, MAIN_METRIC)
            style = architecture_style[hidden_size]

            fig, ax = plt.subplots(figsize=(8.5, 6.0))
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=style["marker"],
                color=style["color"],
                linewidth=1.9,
                capsize=4,
                label=f"Architecture: {hidden_size}",
            )

            set_two_line_title(
                ax,
                f"Inverse noise robustness | Architecture: {hidden_size}",
                f"Species: {species_label}",
            )
            ax.set_xticks(sorted(arch_df["noise_percent"].unique()))
            format_axes(ax)
            ax.legend(fontsize=10, frameon=False)
            fig.tight_layout()

            output_name = f"architecture_{safe_path_token(hidden_size)}"
            save_current_figure(species_dir / output_name)


def plot_by_architecture(df, plots_root):
    """One plot per architecture. Lines compare species configurations."""
    output_dir = Path(plots_root) / "by_architecture"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_species_configs = ordered_unique(df["species_config_name"])
    species_style = {name: style_for(i) for i, name in enumerate(all_species_configs)}

    for hidden_size, arch_df_all in df.groupby("hidden_size", sort=False):
        print(f"\nArchitecture: {hidden_size}")
        print("Number of species combinations:", arch_df_all["species_config_name"].nunique())
        for name in arch_df_all["species_config_name"].drop_duplicates():
            print("  ", name)
        fig, ax = plt.subplots(figsize=(10.5, 7.0))

        for species_config_name, species_df in arch_df_all.groupby("species_config_name", sort=False):
            species_df = species_df.sort_values("noise_percent")
            x = species_df["noise_percent"].to_numpy(dtype=float)
            y, yerr = get_metric_values(species_df, MAIN_METRIC)
            species_label = short_species_label(species_df)
            style = species_style[species_config_name]

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=style["marker"],
                color=style["color"],
                linewidth=1.6,
                capsize=4,
                label=species_label,
            )

        set_two_line_title(
            ax,
            "Inverse noise robustness by species configuration",
            f"Architecture: {hidden_size}",
        )
        ax.set_xticks(sorted(arch_df_all["noise_percent"].unique()))
        format_axes(ax)
        ax.legend(
            title="Input species",
            fontsize=9,
            title_fontsize=10,
            frameon=False,
            ncol=MAX_LEGEND_COLUMNS,
        )
        fig.tight_layout()

        output_base = output_dir / f"architecture_{safe_path_token(hidden_size)}__species_configs"
        save_current_figure(output_base)


def save_plot_manifest(results_root, plots_root, df):
    manifest = {
        "results_root": str(results_root),
        "plots_root": str(plots_root),
        "main_metric": MAIN_METRIC,
        "main_metric_label": MAIN_METRIC_LABEL,
        "use_log_y": USE_LOG_Y,
        "plot_colours": PLOT_COLORS,
        "created_folders": ["by_species", "by_architecture"],
        "overview_folder_created": False,
        "num_species_configurations": int(df["species_config_name"].nunique()),
        "num_architectures": int(df["hidden_size"].nunique()),
        "noise_percent_values": sorted(float(x) for x in df["noise_percent"].unique()),
    }
    pd.Series(manifest).to_json(Path(plots_root) / "plot_manifest.json", indent=4)


# --------------------------------------------------------------------------------------
# Main


def main():
    results_root = get_results_root()
    df = load_aggregate_table(results_root)

    plots_root = results_root / "Plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    plot_by_species(df, plots_root)
    plot_by_architecture(df, plots_root)
    save_plot_manifest(results_root, plots_root, df)

    print(f"Read results from:\n{results_root}")
    print(f"Saved plots to:\n{plots_root}")
    print("Created only these plot folders:")
    print(f"  {plots_root / 'by_species'}")
    print(f"  {plots_root / 'by_architecture'}")


if __name__ == "__main__":
    main()
