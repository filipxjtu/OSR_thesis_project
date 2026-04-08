
# Impaired Dataset `.mat` Artifact Report

**Project:** Deterministic Impaired Signal Dataset Generator  
**Version:** v2  
**Artifact:** `impaired_dataset_v2_seed<dataset_seed>_n<n_per_class>_<mode>.mat`

---

## 1. Objective

To apply deterministic, reproducible impairments to clean signal datasets (classes **0–9**), producing structured MATLAB `.mat` artifacts suitable for:

* Controlled ML training and evaluation
* SNR-aware robustness analysis
* Reproducible degradation studies
* MATLAB–Python interoperability
* Thesis-grade traceability

This report documents the impairment pipeline, structural guarantees, metadata policy, and reproducibility properties of the **v2 impaired dataset generator**.

---

## 2. Function Overview

### Primary Function

```matlab
function impaired_data = generate_impaired_dataset(clean_dataset, spec, mode)
````

### Role

This function is a **batch impairment processor**. It:

1. Accepts a validated clean dataset
2. Verifies canonical clean layout `(N × Ns)`
3. Iterates through each sample deterministically
4. Applies `apply_impairment(x_clean_i, i-1, spec, mode)`
5. Collects full impairment metadata per sample
6. Preserves original labels and clean-generation parameters
7. Builds a versioned impaired artifact
8. Computes artifact hash
9. Saves a deterministic `.mat` file

---

## 3. Dataset Structure

The impaired dataset struct contains:

```matlab
impaired_data.X_imp      % (N × Ns) complex128
impaired_data.y          % (Ns × 1) int32
impaired_data.params     % (Ns × 1 struct)
impaired_data.imp_params % (Ns × 1 struct)
impaired_data.meta       % struct
```
```
Where:
Ns = 10 × n_per_class
N  = spec.N
```

---

## 3.1 Signal Storage Design

### Input

Clean signals arrive in canonical MATLAB layout:

```matlab
X_clean : (N × Ns)
```

Signals are stored as columns.

### Output

Impaired signals are also stored in canonical artifact layout:

```matlab
X_imp : (N × Ns)
```

Each column corresponds to one sample.

### Alignment Rule

For every sample index `i`:

```matlab
X_imp(:,i)  ↔  y(i)  ↔  params(i)  ↔  imp_params(i)
```

This preserves strict one-to-one alignment between:

* impaired signal
* class label
* original clean-generation parameters
* realized impairment parameters

---

## 4. Root Variable and Boundary Semantics

The saved artifact uses the root variable:

```matlab
impaired_data
```

This is part of the MATLAB–Python boundary contract and is version-controlled.

The artifact remains column-oriented at the boundary. Any sample-first handling in Python is an internal consumer-side adaptation only, not a change in artifact meaning.

---

## 5. Deterministic Impairment Model

### Sample Index Mapping

Each sample is impaired using:

```matlab
sample_index = i - 1
```

This matches the clean-generation indexing convention.

### Deterministic Guarantee

Given identical inputs:

* `clean_dataset`
* `spec`
* `mode`

the following must be identical across runs:

```matlab
isequal(impaired1.X_imp, impaired2.X_imp) == true
isequal(impaired1.imp_params, impaired2.imp_params) == true
```

Determinism is therefore inherited from:

```matlab
apply_impairment(x_clean_i, i-1, spec, mode)
```

and from the frozen ordering of samples inside the batch loop.

---

## 6. Impairment Chain

For each sample, impairments are applied in a deterministic sequence.

### Core chain

1. deterministic impairment seed setup
2. input RMS safeguard normalization
3. target SNR selection
4. optional amplitude scaling
5. residual CFO application
6. residual Wiener phase-noise application
7. optional Ricean 2-tap channel application
8. AWGN injection
9. receiver I/Q imbalance
10. receiver DC offset
11. ADC clipping
12. ADC quantization
13. final RMS normalization

### SNR modes

The generator supports:

* `range`
* `fixed`

#### Range mode

* train SNR drawn from `snr_train_db`
* eval SNR drawn from `snr_eval_db`

#### Fixed mode

* single SNR value from `snr_fixed_db`

---

## 7. Residual Realism in v2

Unlike the earlier minimal impairment layer, the v2 impairment path may include controlled residual effects such as:

* carrier frequency offset
* phase noise
* simple channel effects
* delay / echo behavior
* Ricean channel parameterization

These effects are not guessed later by Python; they are explicitly recorded in `imp_params`.

This makes the impaired artifact a **fully traceable degraded signal dataset**, not merely a noisy copy of the clean data.

---

## 8. Validation Layer

Before finalizing the artifact, strict assertions enforce:

* clean input exists and contains required fields
* `X_clean` has shape `(N × Ns)`
* label count matches number of samples
* each returned impaired sample is a column vector
* all output values are finite
* metadata is assembled correctly
* artifact hash is computed after full assembly

Any violation causes immediate failure.

This prevents silent corruption of:

* signal orientation
* sample alignment
* metadata meaning
* reproducibility guarantees

---

## 9. Impairment Parameter Capture

Each sample has a corresponding `imp_params(i)` record.

Minimum v2 fields include:

```matlab
impairment_seed
snr_mode
target_snr_db
realized_snr_db
noise_variance
P_signal
P_noise_realized
amp_scale
rms_out
cfo_hz
phase_noise_std
delay_samp
echo_gain_db
rice_k_db
channel_energy
```

This record preserves both:

* requested impairment settings
* realized numerical effect on the actual sample

---

## 10. Metadata Structure

The impaired artifact inherits clean metadata and extends it.

### Required meta fields include

```matlab
spec_version
dataset_seed
artifact_hash_fn
artifact_hash
layout
dtype_policy
N
Ns
fs
n_per_class
class_set
dataset_version
created_utc
mode
snr_mode
```

### Required values in v2

```matlab
dataset_version  = "impaired_dataset_v2"
layout           = "N_by_Ns_columns_are_samples"
dtype_policy     = "complex128_X_int32_y"
artifact_hash_fn = "simple64_checksum"
```

This ensures direct compatibility with the Python validation layer.

---

## 11. Artifact Hash

The v2 artifact uses:

```matlab
artifact_hash_fn = "simple64_checksum"
```

The hash is computed after the impaired dataset struct is fully assembled.

This provides a deterministic integrity fingerprint over the stored artifact content and prevents silent drift between MATLAB generation and Python loading.

---

## 12. File Artifact

### Filename Format

```matlab
impaired_dataset_v2_seed<dataset_seed>_n<n_per_class>_<mode>.mat
```

Examples:

```matlab
impaired_dataset_v2_seed17_n400_train.mat
impaired_dataset_v2_seed17_n400_eval.mat
```

This encodes:

* artifact version
* deterministic seed identity
* class-count configuration
* operational mode

---

## 13. Storage Format

```matlab
save(filename, 'impaired_data', '-v7.3');
```

### Why `-v7.3`

* HDF5-based storage
* Python compatibility
* large artifact support
* robust struct preservation
* stable cross-language loading

---

## 14. Design Constraints Enforced

The generator does **NOT**:

* modify clean labels
* reorder samples
* alter clean-generation parameters
* perform STFT
* assume model architecture details
* guess missing metadata
* change artifact layout to sample-first form

It strictly applies impairments and assembles the boundary artifact.

---

## 15. Reproducibility Model

The artifact is deterministic if:

```matlab
apply_impairment(x_clean_i, i-1, spec, mode)
```

is deterministic for fixed inputs.

Under this condition:

* impaired signals are reproducible
* impairment metadata is reproducible
* artifact hash is reproducible
* cross-run comparisons are valid

This enables:

* exact experiment regeneration
* robust ablation studies
* stable debugging of training behavior

---

## 16. Contract Summary

The impaired dataset generator satisfies:

| Requirement                          | Status |
| ------------------------------------ | ------ |
| Deterministic impairment             | ✓      |
| Sample-index based reproducibility   | ✓      |
| Canonical `(N × Ns)` artifact layout | ✓      |
| Complex IQ artifact storage          | ✓      |
| Full impairment metadata capture     | ✓      |
| SNR mode traceability                | ✓      |
| Versioned artifact naming            | ✓      |
| Python-validation readiness          | ✓      |
| Hash-based integrity check           | ✓      |

---

## 17. Conceptual Role in Research Pipeline

This artifact represents:

> The controlled degradation layer for the v2 complex-IQ signal system.

It serves as:

* the training/evaluation-facing dataset
* the robustness-testing substrate
* the bridge between clean signal physics and model input generation
* the reproducible source for downstream STFT feature extraction in Python

It separates:

**Clean Signal Physics Layer**
from
**Impairment and Realism Layer**

This separation preserves modularity, interpretability, and thesis-grade experimental control.

---

## 18. Final Role Clarification

In the v2 pipeline:

* **clean artifacts** may cross into Python for validation and traceability
* **impaired artifacts** are the intended model-facing datasets
* Python validates artifact correctness before any training or evaluation step

Thus, this impaired artifact is not just a storage file.

It is the **contracted, reproducible, model-facing representation of controlled signal degradation**.

---
