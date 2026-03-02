# MATLAB-Python Interface Contract

This is a system-level boundary contract between MATLAB (producer) and Python (consumer).

---

## 1. Interface Philosophy

* The `.mat` artifact is treated as a **strict API contract**, not a data dump.
* Python will **never guess**, reshape, cast, or auto-correct.
* Any mismatch is a **hard failure**, not a warning.
* Determinism is enforced through:

  * `dataset_seed`
  * RNG isolation
  * Canonical hashing (`fnv1a64`)
* The artifact hash is part of the reproducibility guarantee.

---

## 2. Artifact Structure is Frozen

### Clean Artifact

Root variable:

```
dataset
```

Fields:

```
X_clean : (N × Ns) float64
y       : (Ns × 1) int32
params  : (Ns × 1 struct)
meta    : struct
```

### Impaired Artifact

Root variable:

```
impaired_data
```

Fields:

```
X_imp      : (N × Ns) float64
y          : (Ns × 1) int32
params     : (Ns × 1 struct)
imp_params : (Ns × 1 struct)
meta       : struct
```

Columns are samples.

This will never change without version bump.

---

## 3. Orientation Is Frozen

Layout:

```
N_by_Ns_columns_are_samples
```

* Signals are stored as columns.
* No transposing in Python.
* Any mismatch is failure.

---

## 4. Dtype Policy Is Frozen

* `X_*` → float64
* `y` → int32
* No implicit casting allowed.
* No float labels.

Future dtype changes require explicit `dtype_policy` update and version bump.

---

## 5. Metadata Is Mandatory

Required keys include:

* `spec_version`
* `dataset_seed`
* `artifact_hash_fn = "fnv1a64"`
* `artifact_hash`
* `layout`
* `dtype_policy`
* `N`
* `Ns`
* `class_set`
* `generator_version`
* `created_utc`

Impaired-only:

* `mode`
* `impairment_version`
* `paired_clean_hash`

Absence of any required key = failure.

---

## 6. Validation Philosophy (Python-Side)

Validation order will be:

1. File structure
2. Required fields
3. Shape check
4. Dtype check
5. Alignment check
6. Metadata consistency check
7. Deterministic hash verification

If any rule fails → raise exception → stop pipeline.

There will be:

* No implicit transpose
* No reshape
* No dtype coercion
* No missing-field fallback
* No warning-and-continue behavior

---

## 7. Failure Conditions Are Explicit

Immediate failure if:

* Shape mismatch
* Dtype mismatch
* NaN/Inf in signals
* Hash mismatch
* Label outside class_set
* Version mismatch (without compatibility rule)
* Missing required metadata
* Multiple competing root structs

---

## 8. Forward Compatibility Is Version-Gated

* Any semantic change requires version bump.
* Python loader will default to strict mode.
* Compatibility must be explicitly declared.
* Adding optional meta keys is allowed.
* Changing layout, dtype, hash canonicalization, or class semantics requires version update.

---

## 9. Contract Freeze Declaration

No implementation will proceed until:

* This contract is stored as `Matlab_Python_Interface_Contract.md`
* It is versioned in the repository

---
