# Dataset Specification v2 (Canonical)

## Status

- dataset_spec_version: v2
- scope: MATLAB-exported datasets consumed by Python
- authority: dataset artifact structure only

---

## File Naming

### Clean
- pattern: `clean_dataset_v{version}_seed{seed}_n{n_per_class}.mat`

### Impaired
- pattern: `impaired_dataset_v{version}_seed{seed}_n{n_per_class}_{mode}.mat`
- mode: `train` | `eval`

---

## Root Variables

### Clean artifact
- root: `dataset`

### Impaired artifact
- root: `impaired_data`

---

## Stored Arrays

### Clean Dataset

| Name    | Shape                     | Dtype      |
|---------|---------------------------|------------|
| X_clean | `(N, Ns)`                 | complex128 |
| y       | `(Ns, 1)`                 | int32      |
| params  | `(Ns, 1)` struct          | struct     |
| meta    | `(1, 1)` struct           | struct     |

### Impaired Dataset

| Name       | Shape             | Dtype      |
|------------|-------------------|------------|
| X_imp      | `(N, Ns)`         | complex128 |
| y          | `(Ns, 1)`         | int32      |
| params     | `(Ns, 1)` struct  | struct     |
| imp_params | `(Ns, 1)` struct  | struct     |
| meta       | `(1, 1)` struct   | struct     |

---

## Layout Rule

- canonical MATLAB storage layout: `N_by_Ns_columns_are_samples`
- columns are samples
- raw dataset storage is column-oriented
- this layout is mandatory at the artifact boundary

### Python handling rule

Python is allowed to transpose **only** in the following case:

- artifact metadata explicitly declares:
  - `layout = "N_by_Ns_columns_are_samples"`
- loaded array arrives or must be handled in sample-first form for internal processing

This transpose is:
- a structural recovery / internal handling step
- not a semantic layout change
- not a general permission for arbitrary reshaping

Any other transpose, reshape, or silent layout correction is forbidden.

---

## Params Semantics

### `params`
- clean generation parameters used to synthesize the signal
- for impaired datasets, `params` must be the original clean-generation parameters for the paired sample

### `imp_params`
Impaired-only metadata describing the applied impairment. Minimum required fields:

- `impairment_seed`
- `snr_mode`
- `target_snr_db`
- `realized_snr_db`
- `noise_variance`
- `P_signal`
- `P_noise_realized`
- `amp_scale`
- `rms_out`
- `cfo_hz`
- `phase_noise_std`
- `delay_samp`
- `echo_gain_db`
- `rice_k_db`
- `channel_energy`


These fields capture the realized impairment state produced by the unified impairment layer:
- `including SNR policy`
- `oscillator effects` 
- `channel effects`
- `post-impairment normalization`

---

## Meta (Required Keys)

Minimum required metadata keys:

- `spec_version`
- `dataset_seed`
- `artifact_hash_fn`
- `artifact_hash`
- `layout`
- `dtype_policy`
- `N`
- `Ns`
- `fs`
- `n_per_class`
- `class_set`
- `dataset_version`
- `created_utc`

### Clean-specific expected values
- `dataset_version = "clean_dataset_v2"`

### Impaired-specific additional required keys
- `mode`
- `snr_mode`

### Required hash policy
- `artifact_hash_fn = "simple64_checksum"`

### Required dtype policy
- `dtype_policy = "complex128_X_int32_y"`

### Required layout policy
- `layout = "N_by_Ns_columns_are_samples"`

---

## Class Set

Known-class artifact class set:

- `class_set = [0,1,2,3,4,5,6,7,8,9]`

---

## Consumer-Side Validation Rules

Python loader must fail if any of the following occur:

- required root variable missing
- multiple competing root variables found
- required arrays missing
- required metadata missing
- shape mismatch
- dtype mismatch
- NaN / Inf in signal arrays
- checksum mismatch
- metadata inconsistency
- invalid `mode`
- invalid `layout`
- invalid `artifact_hash_fn`

### Layout correction exception
The only allowed correction is the explicit MATLAB-column-layout transpose described above.
No other automatic correction is allowed.

---

## Clean vs Impaired Usage Rule

### Clean artifacts
- may cross the MATLAB → Python boundary
- may be loaded for validation, traceability, reporting, and contract verification
- must not be used as training input to the model

### Impaired artifacts
- are the model-facing artifacts for training / evaluation workflows

---

## Notes

- Signals are stored as complex IQ data.
- Dataset structure and artifact meaning are defined here.
- Interface behavior and failure policy are defined by the MATLAB–Python contract.
- Any semantic change to layout, dtype, hash rule, or class meaning requires a new version.