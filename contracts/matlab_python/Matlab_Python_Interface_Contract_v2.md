---
# MATLAB–Python Interface Contract (v2)

#### This is a system-level boundary contract between MATLAB (producer) and Python (consumer).

---

## Changes Introduced in v2

| Change |
|:-------|
| Dtype updated from `float64` → `complex128` |
| `snr_mode` added to metadata |
| Strict dtype enforcement introduced |
| Controlled transpose exception clarified |

---

## 1. Interface Philosophy

* The `.mat` artifact is treated as a **strict API contract**, not a data dump.
* Python will **never guess**, reshape, cast, or auto-correct.
* Any mismatch is a **hard failure**, not a warning.
* Determinism is enforced through:
  * `dataset_seed`
  * RNG isolation
* The artifact hash is part of the reproducibility guarantee.

---

## 2. Artifact Structure is Frozen

### Clean Artifact

```
dataset Fields:
X_clean : (N × Ns) complex128
y       : (Ns × 1) int32
params  : (Ns × 1 struct)
meta    : struct

```

---

### Impaired Artifact

```
impaired_data Fields:
X_imp      : (N × Ns) complex128
y          : (Ns × 1) int32
params     : (Ns × 1 struct)
imp_params : (Ns × 1 struct)
meta       : struct

```
---

### Structural Rule

* Columns are samples
* Layout is fixed and version-controlled
* Any structural modification requires a version bump

---

## 3. Orientation Is Frozen

Layout:

```
N_by_Ns_columns_are_samples

```

### Rules

* Signals are stored as columns
* MATLAB always outputs `(N × Ns)`
* Python must respect this as the **ground truth layout**

---

### Controlled Transpose Exception (v2 Clarification)

A transpose is permitted **only in this case**:

* Metadata explicitly declares:
```

layout = "N_by_Ns_columns_are_samples"

```
* Python internally requires sample-first format `(Ns × N)` for processing

This transpose is:

* A **controlled internal handling step**
* A **structural adaptation**, not a semantic change
* **Not optional outside this case**

---

### Strict Prohibitions

* No arbitrary transpose
* No reshape
* No layout inference
* No silent correction

Any violation → **failure**

---

## 4. Dtype Policy Is Frozen

* `X_*` → `complex128` (**v2 change**)
* `y` → `int32`

### Rules

* No implicit casting
* No precision downgrade
* No real ↔ complex conversion
* No float labels

Any dtype mismatch → **failure**

---

### Dtype Policy Identifier

```
dtype_policy = "complex128_X_int32_y"

```

Future dtype changes require:

* Explicit policy update
* Version bump

---

## 5. Metadata Is Mandatory

### Required Keys

* `spec_version`
* `dataset_seed`
* `artifact_hash_fn = "simple64_checksum"`
* `artifact_hash`
* `layout`
* `dtype_policy`
* `N`
* `Ns`
* `fs`
* `n_per_class`
* `class_set`
* `dataset_version`
* `created_utc`

---

### Impaired-only Keys

* `mode`
* `snr_mode`

---

### Rule

Missing or invalid metadata → **failure**

---

## 6. Validation Philosophy (Python-Side)

Validation order:

1. File structure
2. Required fields
3. Shape check
4. Dtype check
5. Alignment check
6. Metadata consistency check
7. Deterministic hash verification

---

### Enforcement Principles

There will be:

* No implicit transpose (except the defined exception)
* No semantic reshape
* No dtype coercion
* No missing-field fallback
* No warning-and-continue behavior

Failure at any stage → **terminate pipeline**

---

## 7. Failure Conditions Are Explicit

Immediate failure if:

* Shape mismatch
* Dtype mismatch
* NaN or Inf values
* Hash mismatch
* Label outside `class_set`
* Version mismatch (without compatibility rule)
* Missing required metadata
* Multiple root structs detected
* Unauthorized transpose or reshape

---

## 8. Forward Compatibility Is Version-Gated

* Any semantic change requires version bump
* Python operates in strict mode by default
* Compatibility must be explicitly declared

---

### Allowed Without Version Change

* Adding optional metadata fields

---

### Requires Version Change

* Layout changes
* Dtype changes
* Hash method changes
* Class definition changes
* Signal semantics changes

---

## 9. Contract Freeze Declaration

No implementation will proceed until:

* This contract is stored as:
```
Matlab_Python_Interface_Contract_v2.md

```
* It is version-controlled in the repository

---

## Final Note

This contract defines **boundary truth**.

* MATLAB defines **data meaning**
* Python enforces **data correctness**
* No layer is allowed to reinterpret the other

This ensures:

* Reproducibility
* Traceability
* Thesis-grade system integrity

---
