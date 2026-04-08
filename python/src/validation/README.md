
---

## Validation Pipeline — High-Level Algorithm

1. **Locate project root and dataset paths**
   Resolve artifact directories and construct file paths for clean, impaired (train/eval), and unknown datasets.

2. **Load artifacts via contract layer**
   Use `load_artifact(..., load_params=True)` to enforce structure, dtype, orientation, and metadata validity.

3. **Wrap artifacts into typed datasets**
   Convert each artifact into `Dataset(role=...)` for semantic clarity.

4. **Assemble unified dataset bundle**
   Create `DatasetBundle` containing known (clean/train/eval) and unknown (unknown/clean_unk) datasets.

5. **Initialize validation configuration**
   Define expected spec version, class count, and toggles (repro, partial checks, etc.).

6. **Run global validation runner (`validate_all`)**
   Central orchestrator executes all validation stages.

7. **Compute statistical features per dataset**
   Extract time-domain, frequency-domain, and phase-aware metrics via `features.py` and `stats.py`.

8. **Execute validation checks**
   Apply:

   * numeric sanity (NaN/Inf)
   * signal statistics constraints
   * spectral constraints
   * class balance (known only)
   * cross-dataset separation (clean vs impaired, train vs eval)
   * unknown vs known divergence checks

9. **Run reproducibility validation (optional)**
   Recompute metrics across trials → generate stable digests → verify determinism.

10. **Aggregate results into `ValidationSummary`**
    Collect:

    * passed/failed checks
    * metrics
    * thresholds
    * notes

11. **Return summary to caller (no side effects)**
    Neither saving nor raising occurs inside lower layers (runner/gate).

12. **Handle outcome in entry script (`run_validation`)**

    * If **PASS** → save summary JSON report
    * If **FAIL** → print collected failures and raise `ValidationError`

---

## Key Design Properties

* **Single source of truth:** `DatasetBundle`
* **Strict separation:** compute vs decision vs I/O
* **Deterministic validation:** via stable digest
* **Unknown-aware validation:** integrated, not bolted-on
* **Fail-loud policy:** enforced at top level only

---