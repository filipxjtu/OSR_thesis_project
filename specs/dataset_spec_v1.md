# Dataset Specification v1 (Canonical)

## Status
- dataset_spec_version: v1
- scope: MATLAB-exported datasets consumed by Python

## File naming (artifacts/datasets/**)
Clean:
- pattern: clean_dataset_v{version}_seed{seed}.mat
Impaired:
- pattern: impaired_dataset_v{version}_seed{seed}_{mode}.mat
- mode: train | eval

Statistical report naming (reports/statistical/**)
- impaired: impaired_dataset_v{version}_seed{seed}_{mode}_report.md
- clean: clean_dataset_v{version}_seed{seed}_report.md

## Stored arrays
Clean:
- variable_name: X_clean
- shape: (N_samples, N)        <!-- or (N, N_samples) if that is your stored convention -->
- dtype: float64 (MATLAB double)

Impaired:
- variable_name: X_imp
- shape: (N_samples, N)
- dtype: float64

Labels:
- variable_name: y
- shape: (N_samples, 1) or (N_samples,)
- dtype: integer-compatible

## Metadata (required)
- spec_version: v1
- dataset_seed: integer
- generator_version: "v1"
- created_utc: ISO8601 string
- checksum_fnv1a64: uint64 or decimal string
- layout:
  - rows_are_samples: true
  - columns_are_time: true

## Required validation rules (consumer side)
Python loader MUST fail if:
- required variables are missing (X_clean/X_imp, y, metadata)
- shapes mismatch spec
- dtype mismatch (unless explicitly cast rules allow it)
- spec_version mismatch
- checksum mismatch (if checksum is present)

## Notes
- This spec defines *what the dataset means*, the interface contract defines *how it is exchanged and validated*.