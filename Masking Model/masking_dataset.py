from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from sklearn.preprocessing import MaxAbsScaler


@dataclass
class LoadedScalers:
    input_scalers: List[MaxAbsScaler]
    output_scaler: MaxAbsScaler


class InverseKineticsDataset(torch.utils.data.Dataset):
    """
    Dataset for the inverse problem:
        species densities (across all pressure conditions) -> reaction coefficients k.

    Data layout assumptions match the existing KineticLearn project:
    - each raw text file row contains [k_targets..., species_densities...]
    - rows are grouped by pressure condition
    - within each pressure block, the targets are repeated and therefore identical
    """

    def __init__(
        self,
        src_file: str | Path,
        num_species: int,
        num_pressure_conditions: int,
        k_columns: Sequence[int],
        multiply_targets_by: float = 1e30,
        input_scalers: List[MaxAbsScaler] | None = None,
        output_scaler: MaxAbsScaler | None = None,
        max_rows: int | None = None,
        columns: Sequence[int] | None = None,
    ) -> None:
        self.src_file = Path(src_file)
        self.num_species = int(num_species)
        self.num_pressure_conditions = int(num_pressure_conditions)
        self.k_columns = tuple(int(c) for c in k_columns)
        self.multiply_targets_by = float(multiply_targets_by)

        all_data = np.loadtxt(
            self.src_file,
            max_rows=max_rows,
            usecols=columns,
            delimiter=None,
            comments="#",
            skiprows=0,
            dtype=np.float64,
        )
        if all_data.ndim == 1:
            all_data = all_data[None, :]

        n_rows, n_cols = all_data.shape
        if n_rows % self.num_pressure_conditions != 0:
            raise ValueError(
                f"File {self.src_file} has {n_rows} rows, which is not divisible by "
                f"num_pressure_conditions={self.num_pressure_conditions}."
            )

        self.num_targets = len(self.k_columns)
        self.num_cases = n_rows // self.num_pressure_conditions

        density_columns = np.arange(n_cols - self.num_species, n_cols, 1)
        raw_x_flat = all_data[:, density_columns].copy()
        raw_y_flat = all_data[:, self.k_columns].copy()

        x_by_pressure = raw_x_flat.reshape(self.num_pressure_conditions, self.num_cases, self.num_species)
        y_by_pressure = raw_y_flat.reshape(self.num_pressure_conditions, self.num_cases, self.num_targets)

        # Targets are duplicated across pressure rows. Enforce that assumption explicitly.
        y_reference = y_by_pressure[0].copy()
        if not np.allclose(y_by_pressure, y_reference[None, :, :]):
            max_diff = np.max(np.abs(y_by_pressure - y_reference[None, :, :]))
            raise ValueError(
                "Target k values are not identical across pressure blocks, "
                f"but this loader expects them to be repeated. Max difference found: {max_diff}"
            )

        y_scaled_reference = y_reference * self.multiply_targets_by

        self.scalers = LoadedScalers(
            input_scalers=input_scalers or [MaxAbsScaler() for _ in range(self.num_pressure_conditions)],
            output_scaler=output_scaler or MaxAbsScaler(),
        )

        for pressure_idx in range(self.num_pressure_conditions):
            if input_scalers is None:
                self.scalers.input_scalers[pressure_idx].fit(x_by_pressure[pressure_idx])
            x_by_pressure[pressure_idx] = self.scalers.input_scalers[pressure_idx].transform(x_by_pressure[pressure_idx])

        if output_scaler is None:
            self.scalers.output_scaler.fit(y_scaled_reference)
        y_scaled_reference = self.scalers.output_scaler.transform(y_scaled_reference)

        # Final input layout matches the project's existing flattening convention:
        # [pressure_0 species..., pressure_1 species..., ...]
        x_scaled = np.transpose(x_by_pressure, (1, 0, 2)).reshape(self.num_cases, self.num_pressure_conditions * self.num_species)
        x_unscaled = np.transpose(raw_x_flat.reshape(self.num_pressure_conditions, self.num_cases, self.num_species), (1, 0, 2)).reshape(
            self.num_cases, self.num_pressure_conditions * self.num_species
        )

        self.x_data = torch.from_numpy(x_scaled).float()
        self.y_data = torch.from_numpy(y_scaled_reference).float()
        self.x_data_unscaled = torch.from_numpy(x_unscaled).float()
        self.y_data_unscaled = torch.from_numpy(y_reference).float()

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, index: int):
        return self.x_data[index], self.y_data[index]

    def get_data(self):
        return self.x_data, self.y_data

    def get_unscaled_data(self):
        return self.x_data_unscaled, self.y_data_unscaled

    def inverse_transform_targets(self, scaled_targets: np.ndarray) -> np.ndarray:
        original = self.scalers.output_scaler.inverse_transform(scaled_targets)
        return original / self.multiply_targets_by


def species_mask_to_feature_mask(species_mask: np.ndarray, num_pressure_conditions: int) -> np.ndarray:
    species_mask = np.asarray(species_mask, dtype=np.float32)
    if species_mask.ndim != 1:
        raise ValueError("species_mask must be a 1D array")
    return np.tile(species_mask, num_pressure_conditions).astype(np.float32)


def build_feature_names(species_names: Sequence[str], num_pressure_conditions: int) -> list[str]:
    return [
        f"{species_name}_p{pressure_idx + 1}"
        for pressure_idx in range(num_pressure_conditions)
        for species_name in species_names
    ]
