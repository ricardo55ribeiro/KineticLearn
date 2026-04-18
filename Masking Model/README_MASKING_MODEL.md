# Masking Model for the inverse problem

This folder contains a standalone masking-network experiment for the inverse problem:

- **inputs:** species densities (with masking)
- **outputs:** reaction coefficients `k`

## What this model does

A single neural network is trained to handle arbitrary subsets of observed species.
During training:

1. the full species-density vector is loaded from the dataset;
2. a random **species-level mask** is sampled;
3. that same mask is repeated across all pressure conditions;
4. the model receives:
   - the masked density values;
   - the binary mask itself;
5. the model predicts the selected `k` values.

## Important design choices implemented

- **Scheme:** `O2_novib`
- **Species-level masking across all pressures:** yes
- **Training mask distribution:** uniformly from **3** to **N_species** observed species
- **Evaluation:**
  - all species observed
  - exactly 5 species observed (50 seeded random subsets)
  - exactly 3 species observed (50 seeded random subsets)
  - uniformly random from 3 to N species observed (50 seeded random subsets)
- **Architectures tested:**
  - `(30, 30)`
  - `(50, 50)`
  - `(30, 30, 30)`

## How to run

From the `KineticLearn` root:

```bash
python "Masking Model/masking_main.py"
```

## Output structure

Results are saved under:

```text
Results_Masking_Network/O2_novib/uniformly_3_to_11/<timestamp>/
```

with one folder per architecture.

## Files

- `masking_main.py` — main training/evaluation entry point
- `masking_config.py` — experiment and project-root config
- `masking_dataset.py` — dataset loading and scaling
- `masking_model.py` — the MLP model
- `masking_training.py` — mask generation, training, evaluation
- `masking_utils.py` — plotting, saving, summaries
