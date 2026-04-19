from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import importlib.util


@dataclass(frozen=True)
class TrainingConfig:
    scheme: str = "O2_novib"
    activation: str = "tanh"
    architectures: Tuple[Tuple[int, ...], ...] = ((30, 30), (50, 50), (30, 30, 30))
    learning_rate: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 5000
    patience: int = 100
    val_split: float = 0.1
    device: str = "cpu"
    multiply_targets_by: float = 1e30

    # Experiment-level seed for deterministic top-level behavior.
    seed_global: int = 43

    # Independent full training runs, analogous to your FullRun setup.
    seeds: Tuple[int, ...] = tuple(range(32, 52))

    # Masking setup
    min_observed_species_train: int = 3
    num_eval_repeats_per_count: int = 50

    results_root_name: str = "Results_Masking_Network"


@dataclass(frozen=True)
class DatasetConfig:
    scheme: str
    num_pressure_conditions: int
    num_species: int
    species_names: Tuple[str, ...]
    k_columns: Tuple[int, ...]
    main_dataset: Path
    main_dataset_test: Path


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: Path
    training: TrainingConfig
    dataset: DatasetConfig

    @property
    def feature_dim(self) -> int:
        return self.dataset.num_pressure_conditions * self.dataset.num_species

    @property
    def model_input_dim(self) -> int:
        return 2 * self.feature_dim

    @property
    def output_dim(self) -> int:
        return len(self.dataset.k_columns)

    @property
    def experiment_name(self) -> str:
        return (
            f"uniformly_{self.training.min_observed_species_train}"
            f"_to_{self.dataset.num_species}"
        )

    @property
    def evaluation_species_counts(self) -> Tuple[int, ...]:
        return tuple(range(self.training.min_observed_species_train, self.dataset.num_species + 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "training": asdict(self.training),
            "dataset": {
                **asdict(self.dataset),
                "main_dataset": str(self.dataset.main_dataset),
                "main_dataset_test": str(self.dataset.main_dataset_test),
            },
            "feature_dim": self.feature_dim,
            "model_input_dim": self.model_input_dim,
            "output_dim": self.output_dim,
            "experiment_name": self.experiment_name,
            "evaluation_species_counts": list(self.evaluation_species_counts),
        }


def _import_project_dictionary(project_root: Path):
    config_path = project_root / "src" / "config.py"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find project config at: {config_path}")

    spec = importlib.util.spec_from_file_location("kineticlearn_project_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dict


def find_project_root(start_path: Path) -> Path:
    start_path = start_path.resolve()
    candidates: List[Path] = [start_path]
    candidates.extend(start_path.parents)

    for candidate in candidates:
        if (candidate / "src" / "config.py").exists() and (candidate / "data").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the KineticLearn project root. "
        "Expected to find both 'src/config.py' and the 'data/' folder in one of the parent directories."
    )


def build_experiment_config(
    project_root: Path,
    training_cfg: TrainingConfig | None = None,
) -> ExperimentConfig:
    training_cfg = training_cfg or TrainingConfig()
    project_root = project_root.resolve()

    project_dictionary = _import_project_dictionary(project_root)
    if training_cfg.scheme not in project_dictionary:
        available = ", ".join(project_dictionary.keys())
        raise KeyError(f"Scheme '{training_cfg.scheme}' not found in project config. Available: {available}")

    scheme_dict = project_dictionary[training_cfg.scheme]

    dataset_cfg = DatasetConfig(
        scheme=training_cfg.scheme,
        num_pressure_conditions=int(scheme_dict["n_conditions"]),
        num_species=int(scheme_dict["n_densities"]),
        species_names=tuple(scheme_dict["species"]),
        k_columns=tuple(int(c) for c in scheme_dict["k_columns"]),
        main_dataset=(project_root / scheme_dict["main_dataset"]).resolve(),
        main_dataset_test=(project_root / scheme_dict["main_dataset_test"]).resolve(),
    )

    return ExperimentConfig(
        project_root=project_root,
        training=training_cfg,
        dataset=dataset_cfg,
    )
