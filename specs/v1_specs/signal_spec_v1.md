# Signal Specification v1 (Canonical)

## Status
- spec_version: v1
- owner: thesis_project
- scope: synthetic known-class signals (clean + impaired)

## Global sampling
- fs_hz: 10,000,000          <!-- sampling frequency in Hz -->
- N: 4800              <!-- samples per signal -->
- t_definition: n/fs for n=0..N-1

## Signal representation
- real_valued: true
- shape_convention:
  - canonical_vector_shape: (N, 1) in MATLAB functions
  - stored_dataset_shape: (N_samples, N) where each row is one sample  
- dtype:
  - MATLAB: double
  - Python target: float32 after load (unless explicitly stated)

## Known classes
- class_ids: [0,1,2,3,4,5,6]
- class_names:
  - 0: single tone
  - 1: multi tone
  - 2: Linear FM sweep
  - 3: Sinusoidal FM sweep
  - 4: PBN
  - 5: Noise FM
  - 6: Frequency Hopping

## Determinism rules
- dataset_seed: required
- per_sample_rng:
  - depends_on: (dataset_seed, class_id, sample_index)
  - requirement: identical inputs -> identical outputs (signal + params)
- checksum:
  - algorithm: simple64_checksum
  - description: uint64 accumulation over uint8 byte view of
                  [X(:); double(y(:))] with wrap-around behavior
  - scope: checksum computed over stored dataset array + metadata (as defined in contract)

## Normalization
- clean_normalization: none
- impaired_normalization: unit rms
- invariant_goal: scale invariance across SNR

## Notes
- This spec is the canonical reference. If updated, bump to v2 and keep v1 frozen.