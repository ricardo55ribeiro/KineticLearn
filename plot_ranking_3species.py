import ast
from pathlib import Path

import pandas as pd


SCHEME = "O2_novib"
EXPERIMENT_NAME = "FullRun_All3SpeciesCombinations"
FULLRUN_PARENT = Path("Results_NN") / SCHEME / EXPERIMENT_NAME
ANALYSIS_DIR_NAME = "Comparative_Analysis"

ARCHITECTURES = [
    "30, 30",
    "30, 30, 30",
    "50, 50",
]


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


def get_selected_fullrun_dir():
    if not FULLRUN_PARENT.exists():
        raise FileNotFoundError(f"FullRun folder not found: {FULLRUN_PARENT}")

    return FULLRUN_PARENT


def format_combination_from_experiment_name(experiment_name):
    if "__" not in experiment_name:
        return experiment_name

    _, species_part = experiment_name.split("__", 1)

    species_list = []
    for s in species_part.split("_"):
        s = s.strip()
        if not s:
            continue

        if s.upper() == "NONE":
            species_list.append("-")
        else:
            species_list.append(s)

    return "  ".join(species_list)


def summarize_seed_aggregate_df(df, experiment_name):
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

    out = pd.DataFrame()
    out["experiment_name"] = df["experiment_name"] if "experiment_name" in df.columns else experiment_name
    out["combination_of_3_species"] = out["experiment_name"].apply(format_combination_from_experiment_name)
    out["hidden_size_tuple"] = df[hidden_size_col].apply(parse_hidden_size)
    out["architecture"] = out["hidden_size_tuple"].apply(hidden_size_to_label)
    out["num_species_kept"] = pd.to_numeric(df["num_species_kept"], errors="coerce")
    out["test_mse_mean"] = pd.to_numeric(df.get("test_mse_mean"), errors="coerce")
    out["test_mse_std"] = pd.to_numeric(df.get("test_mse_std"), errors="coerce")
    out["test_mse_min"] = pd.to_numeric(df.get("test_mse_min"), errors="coerce")
    out["test_mse_max"] = pd.to_numeric(df.get("test_mse_max"), errors="coerce")
    out["num_seeds"] = pd.NA
    out["source_mode"] = "seed_aggregate"
    return out


def summarize_summary_df(df, experiment_name):
    if "hidden_size" not in df.columns:
        raise RuntimeError(f"summary.csv in {experiment_name} does not contain 'hidden_size'.")

    numeric_cols = ["test_mse", "num_species_kept", "seed"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "num_species_kept" not in df.columns or df["num_species_kept"].isna().all():
        raise RuntimeError(f"Could not determine num_species_kept for {experiment_name} from summary.csv")

    df["hidden_size_tuple"] = df["hidden_size"].apply(parse_hidden_size)
    df["architecture"] = df["hidden_size_tuple"].apply(hidden_size_to_label)
    df["experiment_name"] = df["experiment_name"] if "experiment_name" in df.columns else experiment_name
    df["combination_of_3_species"] = df["experiment_name"].apply(format_combination_from_experiment_name)

    grouped = (
        df.groupby(["experiment_name", "combination_of_3_species", "architecture", "num_species_kept"], as_index=False)
        .agg(
            test_mse_mean=("test_mse", "mean"),
            test_mse_std=("test_mse", "std"),
            test_mse_min=("test_mse", "min"),
            test_mse_max=("test_mse", "max"),
            num_seeds=("seed", "nunique") if "seed" in df.columns else ("test_mse", "size"),
        )
    )
    grouped["source_mode"] = "summary_aggregated"
    return grouped


def load_available_results(fullrun_dir):
    rows = []

    for experiment_dir in sorted(fullrun_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if not experiment_dir.name.startswith("3__"):
            continue

        seed_agg = experiment_dir / "seed_aggregate_summary.csv"
        summary_csv = experiment_dir / "summary.csv"

        if seed_agg.exists():
            df = pd.read_csv(seed_agg)
            rows.append(summarize_seed_aggregate_df(df, experiment_dir.name))
        elif summary_csv.exists():
            df = pd.read_csv(summary_csv)
            rows.append(summarize_summary_df(df, experiment_dir.name))

    if not rows:
        raise RuntimeError(f"No usable experiment folders were found inside: {fullrun_dir}")

    ranking_df = pd.concat(rows, ignore_index=True)
    ranking_df = ranking_df[ranking_df["num_species_kept"].isin([1, 2, 3])].copy()

    ranking_df = ranking_df[ranking_df["architecture"].isin(ARCHITECTURES)].copy()

    if ranking_df.empty:
        raise RuntimeError("No usable 1-, 2-, or 3-species results were found for the requested architectures.")

    ranking_df = ranking_df.sort_values(
        ["architecture", "test_mse_mean", "test_mse_std", "combination_of_3_species"]
    ).reset_index(drop=True)

    return ranking_df


def save_ranking_csv(analysis_dir, ranking_df):
    ranking_df.to_csv(analysis_dir / "three_species_combination_ranking.csv", index=False)


def save_ranking_txt(analysis_dir, ranking_df):
    output_path = analysis_dir / "three_species_combination_ranking.txt"

    lines = []
    lines.append("Ranking of available 3-species combinations by Test MSE")
    lines.append("Sorted from best to worst (lower Test MSE is better).")
    lines.append("")

    for arch in ARCHITECTURES:
        df_arch = ranking_df[ranking_df["architecture"] == arch].copy()
        if df_arch.empty:
            continue

        header_rank = "Rank"
        header_combo = "Combination of 3 Species"
        header_mse = "Test MSE (mean ± std)"
        header_n = "Seeds"

        width_rank = max(len(header_rank), len(str(len(df_arch)))) + 2
        width_combo = max(len(header_combo), df_arch["combination_of_3_species"].map(len).max()) + 4
        width_mse = max(len(header_mse), 28) + 2
        width_n = max(len(header_n), 5) + 2

        total_width = width_rank + width_combo + width_mse + width_n

        lines.append("=" * total_width)
        lines.append(f"Architecture: {arch}")
        lines.append("=" * total_width)
        lines.append(
            f"{header_rank:<{width_rank}}"
            f"{header_combo:<{width_combo}}"
            f"{header_mse:<{width_mse}}"
            f"{header_n:<{width_n}}"
        )
        lines.append("-" * total_width)

        for idx, row in enumerate(df_arch.itertuples(index=False), start=1):
            if pd.isna(row.test_mse_std):
                mse_str = f"{row.test_mse_mean:.6e}"
            else:
                mse_str = f"{row.test_mse_mean:.6e} ± {row.test_mse_std:.6e}"

            seeds_str = "" if pd.isna(row.num_seeds) else str(int(row.num_seeds))

            lines.append(
                f"{idx:<{width_rank}}"
                f"{row.combination_of_3_species:<{width_combo}}"
                f"{mse_str:<{width_mse}}"
                f"{seeds_str:<{width_n}}"
            )

        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def save_summary_txt(analysis_dir, ranking_df, fullrun_dir):
    out_path = analysis_dir / "ranking_build_info.txt"
    lines = [
        f"Source FullRun directory: {fullrun_dir}",
        f"Number of available combinations ranked: {ranking_df['experiment_name'].nunique()}",
        f"Architectures included: {', '.join(ARCHITECTURES)}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    fullrun_dir = get_selected_fullrun_dir()
    analysis_dir = fullrun_dir / ANALYSIS_DIR_NAME
    analysis_dir.mkdir(parents=True, exist_ok=True)

    ranking_df = load_available_results(fullrun_dir)
    save_ranking_csv(analysis_dir, ranking_df)
    save_ranking_txt(analysis_dir, ranking_df)
    save_summary_txt(analysis_dir, ranking_df, fullrun_dir)

    print(f"Ranking files saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
