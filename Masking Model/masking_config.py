from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
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

    seed_global: int = 43
    seed_split: int = 43
    seed_shuffle: int = 43
    seed_weights: int = 43
    seed_training_masks: int = 43
    seed_validation_masks: int = 43
    seed_evaluation_masks: int = 43

    min_observed_species_train: int = 3
    num_eval_repeats: int = 50

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
class EvaluationRegime:
    name: str
    mode: str
    exact_species_count: int | None = None
    repeats: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: Path
    training: TrainingConfig
    dataset: DatasetConfig
    evaluation_regimes: Tuple[EvaluationRegime, ...] = field(default_factory=tuple)

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "training": asdict(self.training),
            "dataset": {
                **asdict(self.dataset),
                "main_dataset": str(self.dataset.main_dataset),
                "main_dataset_test": str(self.dataset.main_dataset_test),
            },
            "evaluation_regimes": [asdict(r) for r in self.evaluation_regimes],
            "feature_dim": self.feature_dim,
            "model_input_dim": self.model_input_dim,
            "output_dim": self.output_dim,
            "experiment_name": self.experiment_name,
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


def build_experiment_config(project_root: Path, training_cfg: TrainingConfig | None = None) -> ExperimentConfig:
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

    evaluation_regimes = (
        EvaluationRegime(name="all_species_observed", mode="all", repeats=1),
        EvaluationRegime(name="exactly_5_species", mode="exact", exact_species_count=5, repeats=training_cfg.num_eval_repeats),
        EvaluationRegime(name="exactly_3_species", mode="exact", exact_species_count=3, repeats=training_cfg.num_eval_repeats),
        EvaluationRegime(
            name=f"uniform_random_{training_cfg.min_observed_species_train}_to_{dataset_cfg.num_species}_species",
            mode="uniform_range",
            exact_species_count=None,
            repeats=training_cfg.num_eval_repeats,
        ),
    )

    return ExperimentConfig(
        project_root=project_root,
        training=training_cfg,
        dataset=dataset_cfg,
        evaluation_regimes=evaluation_regimes,
    )
