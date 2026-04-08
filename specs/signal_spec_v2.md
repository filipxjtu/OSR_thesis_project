# Signal Specification v2 (Canonical)

## Status

- spec_version: v2
- owner: thesis_project
- scope: synthetic signal generation (clean + impaired)
- authority: signal physics, parameter policy, and signal-level semantics

---

## Global Sampling

- `fs_hz = 10,000,000`
- `N = 1024`
- `t_definition = n / fs`, for `n = 0, 1, ..., N-1`
- signal duration = `102.4 µs`

These values are fixed for v2.

---

## Signal Representation

### Domain
- time-domain

### Value type
- complex IQ-valued

### Canonical MATLAB shape
- `(N, 1)`

### Dtype
- MATLAB signal dtype: complex double
- dataset artifact dtype: complex128
- Python may later cast for downstream computation, but artifact truth is complex128

---

## Known Classes

Known synthetic classes for v2:

| ID | Name |
|----|------|
| 0 | Single-Tone Jamming (STJ) |
| 1 | Multi-Tone Jamming (MTJ) |
| 2 | Linear FM Jamming (LFMJ) |
| 3 | Sinusoidal FM Jamming (SFMJ) |
| 4 | Partial Band Noise Jamming (PBNJ) |
| 5 | Frequency Hopping Jamming (FHJ) |
| 6 | OFDM Jamming |
| 7 | Periodic Gaussian Pulse Jamming (PGPJ) |
| 8 | Sliced-Repeating Jamming / ISRJ |
| 9 | Digital False Target Jamming (DFTJ) |

---

## Determinism Rules

### Required global seed
- `dataset_seed` is mandatory

### Per-sample determinism
Each clean sample must be reproducible from:

- `dataset_seed`
- `class_id`
- `sample_index`

Requirement:
- identical inputs must produce identical:
  - signal
  - label
  - parameter record

### RNG isolation
- per-sample RNG must not leak randomness into outer workspace logic
- generator must restore prior RNG state after parameter sampling

---

## Clean Signal Rules

Clean signals are:

- deterministic
- noise-free at the impairment level
- physics-defined by class generator logic
- stored/exported for validation, traceability, and artifact integrity checks

### Important restriction
Clean artifacts may cross into Python for validation and reporting, but they are **not training inputs**.

---

## Impaired Signal Rules

Impaired signals are derived from clean signals.

Properties:
- label is unchanged
- sample pairing with clean source must remain traceable
- impaired signal must preserve length `N`
- impaired output remains complex IQ-valued

---

## Impairment Policy (v2)

Core impairment policy includes:

- SNR-controlled complex AWGN
- optional amplitude scaling
- residual carrier-frequency offset (CFO)
- residual Wiener phase noise
- optional Ricean 2-tap channel effects
- receiver I/Q imbalance
- receiver DC offset
- ADC clipping and quantization
- final RMS normalization after the impairment chain

### SNR modes
- `range`
- `fixed`

#### Range mode
- `snr_train_db = [min, max]`
- `snr_eval_db = [min, max]`

#### Fixed mode
- `snr_fixed_db = scalar`

---

## Normalization Rules

### Clean
- no mandatory dataset-level normalization rule beyond generator-defined waveform normalization
- generator may internally normalize a synthesized waveform to satisfy class design constraints

### Impaired
- final output must be RMS-normalized according to the impairment pipeline

---

## Integrity Constraints

All generated signals must satisfy:

- length = `N`
- finite values only
- complex-valued consistency
- correct class-label consistency
- deterministic reproducibility
- valid parameter record matching active-field policy

---

## Active-Field Policy

Each class activates only the parameter fields relevant to that class.

Rules:
- required active fields must be populated
- non-required scalar numeric fields must remain `NaN`
- nested structured fields are allowed only where class semantics require them

This rule is part of signal-generation correctness.

---

## Clean-to-Impaired Coupling

Some classes may intentionally reuse another clean target generator internally
for physics-consistent DRFM-like behavior.

This is allowed provided:
- coupling is explicit
- same `sample_index` policy is preserved where intended
- the resulting signal remains deterministic

---

## Notes

- This specification defines signal meaning and generation policy, not artifact exchange behavior.
- Dataset storage structure belongs in `dataset_spec_v2.md`.
- Interface enforcement belongs in the MATLAB–Python contract.
- Any change to sampling, class semantics, determinism policy, or signal representation requires a new version.