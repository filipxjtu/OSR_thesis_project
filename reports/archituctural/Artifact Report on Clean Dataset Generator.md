# Clean Dataset `.mat` Artifact Report

**Project:** Deterministic Clean Signal Dataset Generator

**Version:** v1

**Artifact:** `clean_dataset_v1_seed<dataset_seed>.mat`

---

## 1. Objective

To generate a fully deterministic, structured, and validated clean signal dataset for classes **0–6**, stored as a MATLAB `.mat` artifact suitable for:

* Reproducible research
* MATLAB–Python interoperability
* Thesis-grade traceability
* Future impairment extension

This report documents the implementation, structural guarantees, and reproducibility properties of the dataset generator.

---

## 2. Function Overview

### Primary Function

```matlab
function dataset = generate_clean_dataset(n_per_class, spec)
```

#### Role

This function is a **dataset orchestrator**, not a signal generator.
It systematically:

1. Enumerates all classes (0–6)
2. Enumerates all samples per class
3. Assembles outputs from `generate_clean_sample`
4. Enforces structural constraints
5. Validates integrity
6. Saves a deterministic `.mat` artifact

---

## 3. Dataset Structure

The generated dataset struct contains exactly:

```
dataset.X_clean   % N × TotalSamples (double)
dataset.y         % TotalSamples × 1 (int32)
dataset.params    % 1 × TotalSamples struct array
dataset.meta      % struct
```

Where:

```
TotalSamples = 7 × n_per_class
```

---

### 3.1 Signal Storage Design

All signals are stored as:

```
N × 1 column vectors
```

Dataset matrix:

```
X_clean → N × TotalSamples
```

Each column corresponds to:

```
Column i  ↔  y(i)  ↔  params(i)
```

This enforces strict alignment between:

* Signal
* Label
* Parameter metadata

---

## 4. Deterministic Index Mapping

The global sample index is defined as:

```matlab
global_idx = class_id * n_per_class + sample_idx + 1;
```

Loop order is fixed:

```matlab
for class_id = 0:6
    for sample_idx = 0:(n_per_class-1)
```

#### Determinism Guarantee

Given identical `spec` and `n_per_class`:

```
isequal(dataset1.X_clean, dataset2.X_clean) == true
isequal(dataset1.params, dataset2.params) == true
```

No randomization is performed in this function.

---

## 5. Validation Layer

Before saving, strict assertions enforce:

* All values finite
* Signals real-valued
* Correct signal length (`spec.N`)
* Label–sample alignment
* Spec version compliance (`"v1"`)

This prevents silent structural corruption.

---

## 6. Meta Information

The `dataset.meta` struct contains:

```
spec_version
dataset_seed
fs
N
n_per_class
total_samples
creation_time
checksum
```

#### 6.1 Checksum

```
checksum = sum(abs(dataset.X_clean(:)))
```

This provides a lightweight reproducibility fingerprint.

If checksum differs → dataset changed.

---

## 7. File Artifact

### Filename Format

```
clean_dataset_v1_seed<dataset_seed>.mat
```

Example:

```
clean_dataset_v1_seed123.mat
```

This encodes:

* Dataset version
* Deterministic seed identity

---

### Storage Format

```matlab
save(filename, 'dataset', '-v7.3');
```

#### Why `-v7.3`

* HDF5-based backend
* Supports >2GB datasets
* Enables Python compatibility
* Robust for large structured arrays

This ensures future scalability.

---

## 8. Design Constraints Enforced

The function does **NOT**:

* Call `rng`
* Modify `spec`
* Add noise
* Perform STFT
* Add impairments
* Alter signal physics
* Recompute parameters

It strictly assembles and validates.

---

## 9. Reproducibility Model

Determinism depends on:

```
generate_clean_sample(class_id, sample_idx, spec)
```

If that function is deterministic, then:

* Dataset content is deterministic
* File content is deterministic
* Checksum is deterministic

This enables thesis-grade reproducibility.

---

## 10. Contract Summary

The dataset generator satisfies:

| Requirement              | Status |
| ------------------------ | ------ |
| Deterministic order      | ✓      |
| Fixed signal orientation | ✓      |
| Strict index mapping     | ✓      |
| Structured metadata      | ✓      |
| Validation block         | ✓      |
| Versioned artifact       | ✓      |
| Python-ready storage     | ✓      |

---

## 11. Conceptual Role in Research Pipeline

This artifact represents:

> The canonical clean signal baseline.

It serves as:

* Training foundation
* Future impairment input
* Reference dataset for debugging
* Cross-language experiment source

It separates:

**Signal physics layer**
from
**Dataset orchestration layer**

This architectural separation increases modularity and experimental control.

---
