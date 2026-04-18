from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, random_split

from masking_dataset import species_mask_to_feature_mask


@dataclass(frozen=True)
class MaskingConfig:
    num_species: int
    num_pressure_conditions: int
    min_observed_species: int
    max_observed_species: int
    seed: int


class SpeciesMaskGenerator:
    """
    Generates species-level masks and expands them across all pressure blocks.

    The same species mask is repeated for every pressure condition.
    """

    def __init__(self, config: MaskingConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    @property
    def num_species(self) -> int:
        return self.config.num_species

    @property
    def feature_dim(self) -> int:
        return self.config.num_species * self.config.num_pressure_conditions

    def sample_species_mask(self, observed_species_count: int | None = None) -> np.ndarray:
        if observed_species_count is None:
            observed_species_count = int(
                self.rng.integers(
                    self.config.min_observed_species,
                    self.config.max_observed_species + 1,
                )
            )
        observed_species_count = int(observed_species_count)
        if not (0 <= observed_species_count <= self.config.num_species):
            raise ValueError(
                f"observed_species_count={observed_species_count} is outside the valid range [0, {self.config.num_species}]"
            )

        observed_indices = self.rng.choice(self.config.num_species, size=observed_species_count, replace=False)
        species_mask = np.zeros(self.config.num_species, dtype=np.float32)
        species_mask[observed_indices] = 1.0
        return species_mask

    def sample_feature_mask(self, observed_species_count: int | None = None) -> np.ndarray:
        species_mask = self.sample_species_mask(observed_species_count=observed_species_count)
        return species_mask_to_feature_mask(species_mask, self.config.num_pressure_conditions)

    def sample_batch_feature_masks(self, batch_size: int, observed_species_count: int | None = None) -> np.ndarray:
        masks = [self.sample_feature_mask(observed_species_count=observed_species_count) for _ in range(batch_size)]
        return np.stack(masks, axis=0).astype(np.float32)


def concat_masked_inputs(x_scaled: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
    masked_x = x_scaled * feature_mask
    return torch.cat([masked_x, feature_mask], dim=1)


@dataclass
class TrainingArtifacts:
    model_state_dict: Dict[str, Any]
    history: Dict[str, List[float]]
    train_subset_indices: List[int]
    val_subset_indices: List[int]


@dataclass
class EvaluationArtifacts:
    regime_name: str
    repeats: int
    sampled_species_subsets: List[List[str]]
    observed_species_counts: List[int]
    predictions_scaled_repeats: np.ndarray  # [repeats, samples, outputs]
    predictions_scaled_mean: np.ndarray     # [samples, outputs]
    predictions_scaled_std: np.ndarray      # [samples, outputs]
    predictions_unscaled_mean: np.ndarray   # [samples, outputs]
    predictions_unscaled_std: np.ndarray    # [samples, outputs]
    metrics: Dict[str, Any]


class FixedMaskValidationDataset(torch.utils.data.Dataset):
    def __init__(self, base_subset: Subset, full_feature_masks: np.ndarray):
        if len(base_subset) != len(full_feature_masks):
            raise ValueError(
                f"Mask matrix length ({len(full_feature_masks)}) must match subset length ({len(base_subset)})."
            )
        self.base_subset = base_subset
        self.full_feature_masks = torch.from_numpy(full_feature_masks).float()

    def __len__(self) -> int:
        return len(self.base_subset)

    def __getitem__(self, idx: int):
        x_scaled, y_scaled = self.base_subset[idx]
        return x_scaled, y_scaled, self.full_feature_masks[idx]


class IdentityMaskedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        return self.base_dataset[idx]


def train_masked_model(
    model: torch.nn.Module,
    dataset_train: torch.utils.data.Dataset,
    masking_config: MaskingConfig,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    val_split: float,
    device: str,
    split_seed: int,
    shuffle_seed: int,
    training_mask_seed: int,
    validation_mask_seed: int,
) -> TrainingArtifacts:
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_len = int((1.0 - val_split) * len(dataset_train))
    val_len = len(dataset_train) - train_len

    split_generator = torch.Generator().manual_seed(split_seed)
    train_subset, val_subset = random_split(dataset_train, [train_len, val_len], generator=split_generator)

    # Fixed validation masks for stable early stopping.
    val_mask_generator = SpeciesMaskGenerator(
        MaskingConfig(
            num_species=masking_config.num_species,
            num_pressure_conditions=masking_config.num_pressure_conditions,
            min_observed_species=masking_config.min_observed_species,
            max_observed_species=masking_config.max_observed_species,
            seed=validation_mask_seed,
        )
    )
    val_full_feature_masks = val_mask_generator.sample_batch_feature_masks(batch_size=len(val_subset))
    val_masked_dataset = FixedMaskValidationDataset(val_subset, val_full_feature_masks)

    shuffle_generator = torch.Generator().manual_seed(shuffle_seed)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=shuffle_generator)
    val_loader = DataLoader(val_masked_dataset, batch_size=batch_size, shuffle=False)

    training_mask_generator = SpeciesMaskGenerator(
        MaskingConfig(
            num_species=masking_config.num_species,
            num_pressure_conditions=masking_config.num_pressure_conditions,
            min_observed_species=masking_config.min_observed_species,
            max_observed_species=masking_config.max_observed_species,
            seed=training_mask_seed,
        )
    )

    best_model_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    model.to(device)

    for _epoch in range(max_epochs):
        model.train()
        train_loss_accumulator = 0.0

        for x_scaled, y_scaled in train_loader:
            x_scaled = x_scaled.to(device)
            y_scaled = y_scaled.to(device)

            batch_feature_masks_np = training_mask_generator.sample_batch_feature_masks(batch_size=x_scaled.shape[0])
            batch_feature_masks = torch.from_numpy(batch_feature_masks_np).float().to(device)
            model_inputs = concat_masked_inputs(x_scaled, batch_feature_masks)

            optimizer.zero_grad()
            predictions = model(model_inputs)
            loss = criterion(predictions, y_scaled)
            loss.backward()
            optimizer.step()

            train_loss_accumulator += float(loss.item()) * x_scaled.shape[0]

        model.eval()
        val_loss_accumulator = 0.0
        with torch.no_grad():
            for x_scaled, y_scaled, fixed_feature_mask in val_loader:
                x_scaled = x_scaled.to(device)
                y_scaled = y_scaled.to(device)
                fixed_feature_mask = fixed_feature_mask.to(device)
                model_inputs = concat_masked_inputs(x_scaled, fixed_feature_mask)
                predictions = model(model_inputs)
                loss = criterion(predictions, y_scaled)
                val_loss_accumulator += float(loss.item()) * x_scaled.shape[0]

        train_loss = train_loss_accumulator / len(train_subset)
        val_loss = val_loss_accumulator / len(val_subset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_model_state)
    return TrainingArtifacts(
        model_state_dict=best_model_state,
        history=history,
        train_subset_indices=list(train_subset.indices),
        val_subset_indices=list(val_subset.indices),
    )


def _predict_with_fixed_feature_mask(
    model: torch.nn.Module,
    x_scaled: torch.Tensor,
    feature_mask_np: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    feature_mask_tensor = torch.from_numpy(feature_mask_np.astype(np.float32))

    with torch.no_grad():
        for start_idx in range(0, len(x_scaled), batch_size):
            end_idx = min(start_idx + batch_size, len(x_scaled))
            x_batch = x_scaled[start_idx:end_idx].to(device)
            repeated_feature_mask = feature_mask_tensor.unsqueeze(0).repeat(x_batch.shape[0], 1).to(device)
            model_inputs = concat_masked_inputs(x_batch, repeated_feature_mask)
            batch_predictions = model(model_inputs).cpu().numpy()
            predictions.append(batch_predictions)

    return np.concatenate(predictions, axis=0)


def evaluate_regime(
    model: torch.nn.Module,
    x_scaled: torch.Tensor,
    y_scaled: torch.Tensor,
    inverse_transform_targets,
    regime_name: str,
    regime_mode: str,
    species_names: Sequence[str],
    num_pressure_conditions: int,
    min_observed_species_train: int,
    repeats: int,
    seed: int,
    device: str,
    exact_species_count: int | None = None,
) -> EvaluationArtifacts:
    num_species = len(species_names)
    mask_generator = SpeciesMaskGenerator(
        MaskingConfig(
            num_species=num_species,
            num_pressure_conditions=num_pressure_conditions,
            min_observed_species=min_observed_species_train,
            max_observed_species=num_species,
            seed=seed,
        )
    )

    if regime_mode == "all":
        sampled_feature_masks = [np.ones(num_species * num_pressure_conditions, dtype=np.float32)]
        sampled_species_masks = [np.ones(num_species, dtype=np.float32)]
        repeats = 1
    elif regime_mode == "exact":
        if exact_species_count is None:
            raise ValueError("exact_species_count must be provided for regime_mode='exact'")
        sampled_species_masks = [mask_generator.sample_species_mask(observed_species_count=exact_species_count) for _ in range(repeats)]
        sampled_feature_masks = [species_mask_to_feature_mask(mask, num_pressure_conditions) for mask in sampled_species_masks]
    elif regime_mode == "uniform_range":
        sampled_species_masks = [mask_generator.sample_species_mask(observed_species_count=None) for _ in range(repeats)]
        sampled_feature_masks = [species_mask_to_feature_mask(mask, num_pressure_conditions) for mask in sampled_species_masks]
    else:
        raise ValueError(f"Unsupported regime_mode '{regime_mode}'")

    predictions_scaled_repeats = []
    sampled_species_subsets: list[list[str]] = []
    observed_species_counts: list[int] = []
    for feature_mask, species_mask in zip(sampled_feature_masks, sampled_species_masks):
        predictions_scaled = _predict_with_fixed_feature_mask(
            model=model,
            x_scaled=x_scaled,
            feature_mask_np=feature_mask,
            device=device,
        )
        predictions_scaled_repeats.append(predictions_scaled)
        observed_indices = np.flatnonzero(species_mask > 0.5)
        sampled_species_subsets.append([species_names[idx] for idx in observed_indices])
        observed_species_counts.append(int(len(observed_indices)))

    predictions_scaled_repeats_np = np.stack(predictions_scaled_repeats, axis=0)
    predictions_scaled_mean = predictions_scaled_repeats_np.mean(axis=0)
    predictions_scaled_std = predictions_scaled_repeats_np.std(axis=0, ddof=0)

    predictions_unscaled_repeats_np = np.stack(
        [inverse_transform_targets(pred) for pred in predictions_scaled_repeats_np],
        axis=0,
    )
    predictions_unscaled_mean = predictions_unscaled_repeats_np.mean(axis=0)
    predictions_unscaled_std = predictions_unscaled_repeats_np.std(axis=0, ddof=0)

    y_scaled_np = y_scaled.numpy()
    y_unscaled_np = inverse_transform_targets(y_scaled_np)

    mse_scaled_per_repeat = [mean_squared_error(y_scaled_np, pred) for pred in predictions_scaled_repeats_np]
    mse_unscaled_per_repeat = [mean_squared_error(y_unscaled_np, pred) for pred in predictions_unscaled_repeats_np]

    metrics: Dict[str, Any] = {
        "regime_name": regime_name,
        "regime_mode": regime_mode,
        "repeats": repeats,
        "observed_species_count_mean": float(np.mean(observed_species_counts)),
        "observed_species_count_std": float(np.std(observed_species_counts, ddof=0)),
        "mse_scaled_mean_over_repeats": float(np.mean(mse_scaled_per_repeat)),
        "mse_scaled_std_over_repeats": float(np.std(mse_scaled_per_repeat, ddof=0)),
        "rmse_scaled_mean_over_repeats": float(np.sqrt(np.mean(mse_scaled_per_repeat))),
        "mse_unscaled_mean_over_repeats": float(np.mean(mse_unscaled_per_repeat)),
        "mse_unscaled_std_over_repeats": float(np.std(mse_unscaled_per_repeat, ddof=0)),
        "rmse_unscaled_mean_over_repeats": float(np.sqrt(np.mean(mse_unscaled_per_repeat))),
        "mse_scaled_of_mean_prediction": float(mean_squared_error(y_scaled_np, predictions_scaled_mean)),
        "mse_unscaled_of_mean_prediction": float(mean_squared_error(y_unscaled_np, predictions_unscaled_mean)),
    }
    metrics["rmse_scaled_of_mean_prediction"] = float(np.sqrt(metrics["mse_scaled_of_mean_prediction"]))
    metrics["rmse_unscaled_of_mean_prediction"] = float(np.sqrt(metrics["mse_unscaled_of_mean_prediction"]))

    return EvaluationArtifacts(
        regime_name=regime_name,
        repeats=repeats,
        sampled_species_subsets=sampled_species_subsets,
        observed_species_counts=observed_species_counts,
        predictions_scaled_repeats=predictions_scaled_repeats_np,
        predictions_scaled_mean=predictions_scaled_mean,
        predictions_scaled_std=predictions_scaled_std,
        predictions_unscaled_mean=predictions_unscaled_mean,
        predictions_unscaled_std=predictions_unscaled_std,
        metrics=metrics,
    )
