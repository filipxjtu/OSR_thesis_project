# System Specification v2 (Canonical)

## Status

- system_spec_version: v2
- scope: end-to-end MATLAB → Python → model pipeline
- authority: system-level processing rules

---

## Sampling and Length

- `fs = 10,000,000 Hz`
- `N = 1024 samples`
- duration = `102.4 µs`

All modules in the pipeline must respect these values.

---

## Raw Signal Representation

### MATLAB producer side
- domain: time-domain
- type: complex IQ-valued
- canonical shape: `(N, 1)`

### Dataset artifact boundary
- storage layout: `(N, Ns)`
- columns are samples
- dtype: complex128

### Python consumer side
Python may internally use sample-first handling when needed, but only under the controlled transpose exception:

- metadata explicitly declares `layout = "N_by_Ns_columns_are_samples"`
- transpose is performed only to internally handle sample-major processing

This is an implementation accommodation, not a contract relaxation.

---

## MATLAB → Python Boundary Rule

The `.mat` artifact is a strict boundary object.

Allowed:
- explicit validation
- explicit metadata-driven transpose for the column-layout case above

Not allowed:
- arbitrary reshape
- silent layout guessing
- silent dtype coercion
- semantic reinterpretation of artifact contents

---

## STFT Representation

Model input is built in Python from raw complex IQ signals.

### Transformation
- Short-Time Fourier Transform (STFT)

### Fixed STFT parameters
- window: `Hann`
- window_length: `128`
- hop_length: `64`
- n_fft: `128`
- spectrum: full FFT
- fft_type: two-sided
- padding: none
- centering: disabled

### STFT output
- magnitude is computed from STFT
- log compression is applied:

  `X_tf = log(1 + |STFT(x)|)`

- global mean normalization is then applied:

  `X_tf <- X_tf / mean(X_tf)`

### Output shape
- `(n_fft, time_frames)`

---

## Fixed Processing Order

For impaired model-facing signals, the fixed order is:

1. clean signal generation
2. per-sample deterministic impairment seeding
3. input RMS safeguard normalization
4. target SNR selection
5. optional amplitude scaling
6. residual CFO / phase-noise application
7. optional Ricean 2-tap channel application
8. AWGN injection
9. receiver front-end impairment application
10. ADC clipping and quantization
11. final RMS normalization
12. STFT
13. magnitude extraction
14. log compression
15. global mean normalization

---

## Normalization Rules

### Time-domain
- applied to impaired signals at the end of the impairment chain
- rule: unit RMS normalization

### Time-frequency domain
- applied after STFT magnitude + log compression
- rule: global mean normalization
- not per-frame normalization

---

## Dataset Usage Rules

### Training
- impaired known-class synthetic data only

### Validation
- may use impaired known-class data
- may use additional unknown-class data when the experiment requires open-set evaluation
- clean data may be loaded for validation checks, not for model fitting

### Testing / inference
- may include:
  - known synthetic
  - unknown synthetic
  - real known / unknown data, depending on experiment stage

---

## Validation Responsibility

Python must validate, not assume.

Validation includes:
- root detection
- required fields
- shape
- dtype
- metadata consistency
- checksum verification
- layout enforcement
- mode validation
- numeric-domain checks

### Layout exception
The only allowed structural correction is the explicit transpose used to handle MATLAB column-oriented storage in Python.

---

## Artifact Truth vs Internal Representation

Artifact truth:
- signals are stored as `(N, Ns)` complex128
- labels are `(Ns, 1)` int32
- metadata defines meaning

Internal Python representation may differ only for implementation convenience, provided:
- metadata is honored
- semantic meaning is preserved
- no uncontrolled correction occurs

---

## Contract Dependencies

This system spec depends on:

- `signal_spec_v2.md`
- `dataset_spec_v2.md`
- MATLAB–Python interface contract

---

## Notes

- STFT parameters are part of the system definition.
- Artifact layout is part of the system definition.
- Complex IQ representation is part of the system definition.
- Any change to sampling, STFT, normalization, layout semantics, or dtype policy requires a new version.