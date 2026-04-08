
# Clean Dataset `.mat` Artifact Report

**Project:** Deterministic Clean Signal Dataset Generator  
**Version:** v2  
**Artifact:** `clean_dataset_v2_seed<dataset_seed>_n<n_per_class>.mat`

---

## 1. Objective

To generate a fully deterministic, structured, and validated clean signal dataset for classes **0–9**, stored as a MATLAB `.mat` artifact suitable for:

* Reproducible research  
* MATLAB–Python interoperability  
* Thesis-grade traceability  
* Controlled impairment pairing  
* Validation-layer verification (pre-training)  

This report documents the implementation, structural guarantees, and reproducibility properties under **v2 specifications**.

---

## 2. Function Overview

### Primary Function

```matlab
function dataset = generate_clean_dataset(n_per_class, spec)
````

### Role

This function is a **dataset orchestrator**, not a signal generator.

It:

1. Enumerates all classes (0–9)
2. Enumerates all samples per class
3. Calls `generate_clean_sample`
4. Assembles outputs into canonical structure
5. Enforces structural constraints
6. Validates integrity
7. Computes artifact hash
8. Saves a deterministic `.mat` artifact

---

## 3. Dataset Structure

The dataset struct contains exactly:

```
dataset.X_clean   % (N × Ns) complex128
dataset.y         % (Ns × 1) int32
dataset.params    % (Ns × 1 struct)
dataset.meta      % struct
```

Where:

```
Ns = 10 × n_per_class
```

---

## 4. Signal Storage Design

All signals are:

```
(N × 1) column vectors (complex IQ)
```

Dataset matrix:

```
X_clean → (N × Ns)
```

Mapping:

```
Column i  ↔  y(i)  ↔  params(i)
```

### Layout (Frozen)

```
layout = "N_by_Ns_columns_are_samples"
```

---

## 5. Deterministic Index Mapping

Global index:

```matlab
global_idx = class_id * n_per_class + sample_idx + 1;
```

Loop order:

```matlab
for class_id = 0:9
    for sample_idx = 0:(n_per_class-1)
```

### Determinism Guarantee

Given identical inputs:

```
isequal(dataset1.X_clean, dataset2.X_clean) == true
isequal(dataset1.params, dataset2.params) == true
```

No randomness is initiated at the dataset-orchestration level.
Dataset content determinism is inherited from deterministic per-sample generation under fixed `dataset_seed`, `class_id`, and `sample_index`.

---

## 6. Validation Layer

Strict assertions enforce:

* No NaN / Inf
* Complex dtype (`complex128`)
* Signal length = `spec.N`
* Column alignment
* Label consistency
* Metadata completeness
* Spec version = `"v2"`

Failure → immediate termination.

---

## 7. Metadata Structure

`dataset.meta` includes:

```
spec_version        = "v2"
dataset_seed
artifact_hash_fn    = "simple64_checksum"
artifact_hash
layout              = "N_by_Ns_columns_are_samples"
dtype_policy        = "complex128_X_int32_y"
fs
N
Ns
n_per_class
class_set
dataset_version     = "clean_dataset_v2"
created_utc
```

---

## 8. Artifact Hash (v2)

Hash replaces checksum.

Definition:

```
artifact_hash = simple64_checksum([real(X(:)); imag(X(:)); double(y(:))])
```


artifact_hash is computed by the project’s `simple64_checksum` implementation over the complex signal content and labels.

**Properties:**

* Deterministic
* Cross-language consistent
* Order-sensitive
* Required for validation

---

## 9. File Artifact

### Filename

```
clean_dataset_v2_seed<seed>_n<n_per_class>.mat
```

Example:

```
clean_dataset_v2_seed123_n400.mat
```

---

### Storage Format

```matlab
save(filename, 'dataset', '-v7.3');
```

Reason:

* HDF5 backend
* Python compatibility
* Large dataset support

---

## 10. Design Constraints

The function does **NOT**:

* Call `rng`
* Modify `spec`
* Add noise
* Perform STFT
* Apply impairments
* Alter signal physics
* Recompute parameters

It strictly orchestrates and validates.

---

## 11. Reproducibility Model

Determinism depends on:

```
generate_clean_sample(class_id, sample_idx, spec)
```

If deterministic → entire dataset deterministic:

* Content
* Ordering
* Hash

---

## 12. Contract Alignment

This artifact fully complies with:

* MATLAB–Python Interface Contract (v2)
* Column-based layout enforcement
* Complex dtype policy
* Metadata completeness
* Hash verification requirement

---

## 13. Conceptual Role

This dataset represents:

> The canonical clean signal baseline (v2, complex IQ domain)

It is used for:

* Validation benchmarking
* Impairment generation input
* Debugging reference
* Cross-language verification

### Critical Rule

Clean data:

* **Allowed to cross boundary (for validation)**
* **Strictly prohibited from model training**

---

## 14. System Position

This module separates:

```
Signal Physics Layer
        ↓
Clean Dataset (this artifact)
        ↓
Impairment Layer
        ↓
Feature Extraction (STFT)
        ↓
Model Training
```

This separation ensures:

* Modularity
* Traceability
* Experimental control
* Thesis-grade rigor
---