# Impaired Dataset `.mat` Artifact Report

**Project:** Deterministic Impaired Signal Dataset Generator

**Version:** v1

**Artifact:** `impaired_dataset_v1_seed<dataset_seed>_<mode>.mat`

---

## 1. Objective

To apply deterministic, reproducible impairments to clean signal datasets (classes **0–6**), producing structured `.mat` artifacts suitable for:

* Robust ML training with controlled degradation
* SNR-aware model evaluation
* Ablation studies on impairment effects
* Cross-validation with fixed impairment patterns
* MATLAB–Python interoperability

This report documents the impairment pipeline, structural guarantees, and reproducibility properties.

---

## 2. Function Overview

### Primary Function

```matlab
function impaired = generate_impaired_dataset(clean_dataset, spec, mode)
```

#### Role

This function is a **batch impairment processor**. It:

1. Takes a validated clean dataset
2. Enforces consistent orientation (`N × Nsamples`)
3. Iterates through each sample deterministically
4. Applies `apply_impairment` with sample_index = i-1
5. Collects impairment parameters
6. Preserves original labels and metadata
7. Saves deterministic impaired `.mat` artifact

---

## 3. Dataset Structure

The impaired dataset struct contains:

```
impaired.X_imp       % Nsamples × N (double) - ML ready (samples as rows)
impaired.y           % Nsamples × 1 (int32)
impaired.params      % Original clean parameters (Nsamples × 1 struct)
impaired.imp_params  % Impairment parameters (Nsamples × 1 struct)
impaired.meta        % struct
impaired.X_clean     % (optional) if spec.keep_X_clean == true
```

Where:

```
Nsamples = 7 × n_per_class (from clean dataset)
N = spec.N (signal length)
```

---

### 3.1 Signal Storage Design

**Input:** Clean signals stored as `N × Nsamples` (signals in columns)

**Output:** Impaired signals stored as `Nsamples × N` (samples as rows)

This transformation enables:

* Direct use in ML frameworks (samples as rows, features as columns)
* Easy export to CSV/tables
* Compatibility with MATLAB's fitcnet, fitcecoc, etc.
* Intuitive indexing: `X_imp(i,:)` = i-th sample

---

## 4. Deterministic Impairment Model

### Sample Index Mapping

```matlab
sample_index = i-1  % 0-based indexing consistent with clean generation
```

Each sample `i` receives impairment seed:

```matlab
base = uint32(mod(spec.dataset_seed, 2^32));
idx  = uint32(mod(i-1, 2^32));
imp_seed = double(bitxor(base, uint32(1664525) * idx + uint32(1013904223)));
```

#### Determinism Guarantee

Given identical `clean_dataset`, `spec`, and `mode`:

```
isequal(impaired1.X_imp, impaired2.X_imp) == true
isequal(impaired1.imp_params, impaired2.imp_params) == true
```

All randomness (SNR draw, amplitude scaling, phase offset) is seed-controlled.

---

## 5. Impairment Chain

For each sample, in order:

1. **SNR Selection** – Uniform draw from `snr_train_db` or `snr_eval_db`
2. **Amplitude Scaling** (optional) – Scale signal by random factor in `amp_scale_range`
3. **Phase Offset** (optional) – Real-safe proxy: polarity flip if `cos(phase) < 0`
4. **AWGN Addition** – Noise power = `P_signal * 10^(-target_snr_db/10)`
5. **RMS Normalization** – Output = `x_awgn / sqrt(mean(x_awgn.^2))`

All impairments are applied **before** final normalization.

---

## 6. Validation Layer

Before saving, strict assertions enforce:

* All output values finite
* Clean dataset contains required fields
* Signal length matches `spec.N`
* Label count matches sample count
* Impairment parameters collected for all samples

This ensures downstream ML pipelines receive clean, validated data.

---

## 7. Meta Information

The `impaired.meta` struct extends clean metadata with:

```
mode                    % "train" or "eval"
has_clean_link         % true
N                      % signal length
N_samples              % number of samples
dataset_seed           % from spec
version                 % "impaired_dataset_v1"
+ all fields from clean_dataset.meta
```

This preserves traceability back to the source clean dataset.

---

## 8. File Artifact

### Filename Format

```
impaired_dataset_v1_seed<dataset_seed>_<mode>.mat
```

Examples:

```
impaired_dataset_v1_seed42_train.mat
impaired_dataset_v1_seed42_eval.mat
```

This encodes:

* Dataset version
* Deterministic seed identity
* Operational mode (train/eval)

---

### Storage Format

```matlab
save(filename, 'impaired', '-v7.3');
```

#### Why `-v7.3`

* HDF5-based backend
* Supports >2GB datasets
* Python compatibility (h5py)
* Preserves struct arrays
* Future-proof for larger datasets

---

## 9. Design Constraints Enforced

The function does **NOT**:

* Modify clean dataset
* Reorder samples
* Change labels
* Alter original parameters
* Introduce non-determinism
* Assume any specific ML framework

It strictly impairs and assembles.

---

## 10. Reproducibility Model

Determinism depends on:

```matlab
apply_impairment(x_clean_i, i-1, spec, mode)
```

If that function is deterministic, then:

* Impaired dataset content is deterministic
* Impairment parameters are deterministic
* File content is deterministic

This enables:

* Exact replication of experiments
* Comparison across impairment strategies
* Debugging with fixed noise patterns

---

## 11. Contract Summary

The impaired dataset generator satisfies:

| Requirement                   | Status |
| ----------------------------- | ------ |
| Deterministic impairment      | ✓      |
| Sample-index based seeding    | ✓      |
| ML-ready orientation          | ✓      |
| SNR traceability              | ✓      |
| Full parameter capture        | ✓      |
| Clean signal link (optional)  | ✓      |
| Versioned artifact            | ✓      |
| Python-ready storage          | ✓      |

---

## 12. Conceptual Role in Research Pipeline

This artifact represents:

> The controlled degradation layer.

It serves as:

* Training/evaluation datasets with known SNR
* Benchmark for noise robustness
* Input for ablation studies
* Cross-language experiment source
* Extension point for new impairments

It separates:

**Clean signal physics layer**
from
**Impairment layer**

This architectural separation enables systematic study of model robustness across controlled degradation types and levels.