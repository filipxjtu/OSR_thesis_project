Sampling & Length

- fs = 10,000,000 Hz
- N = 4800 samples per class
- duration = 0.00048 sec

* All signals generated, stored, and validated must match these values exactly


Signal Domain and Representation

* Raw Signal
 - Domain: Time domain
 - Type: Real-valued
 - Shape: (N, )

* Model Input Representation
 - Transformation: Short-Time Fourier Transform (STFT)
 - STFT is applied only in Python, never in MATLAB.
 - Raw signals are stored and exchanged before any transformation.


STFT Specification

* All STFT parameters are fixed as follows:
 - Window type: Hann
 - Window length: 256 samples
 - Hop size: 128 samples (50% overlap)
 - FFT size: 256
 - Spectrum: One-sided
 - Padding: None
 - Centering: Disabled

* STFT Output
 - Magnitude spectrum is computed
 - Log compression applied as:
        𝑋_𝑡𝑓 = log (1 + ∣𝑆𝑇𝐹𝑇 (𝑥)∣)
 - Resulting tensor is the model input

* No deviation from these parameters is allowed without a new specification version.


Amplitude & Normalization rules

* Raw Signal Normalization
 - Applied after impairment, before STFT
 - Rule: Unit average power normalization
 - Formula:
     𝑥 ← 𝑥 / {sqrt(E[𝑥^2])}​
 - Applied per signal, not globally

* Time-Frequency Normalization
 - Applied after STFT
 - Rule: Window-wise normalization
 - Each time frame is normalized by its mean magnitude

*Order is fixed:
 - Impairment → Power normalization → STFT → Log compression → Window normalization


D. Class Semantics 

** KNOWN CLASSES (k-mat) 
| Class ID |  Jamming Type 
| -------- |  ------------------------------------------------
|    0     |  Single-Tone Jamming (STJ / CWJ)
|    1     |  Multi-Tone Jamming (MTJ)
|    2     |  Linear Frequency Modulation Jamming (LFMJ)
|    3     |  Sinusoidal FM Jamming (SFMJ)
|    4     |  Partial Band Noise Jamming (PBNJ / PBJ)
|    5     |  Noise FM Jamming (NFMJ / FMN)
|    6     |  Frequency-Hopping Jamming (FHJ) 

** UNKNOWN CLASSES (u-mat)
 - Pulse / Periodic Pulse Jamming (PJ / PPNJ)
 - Comb Spectrum Jamming (CSJ)   
 - Interrupted Sampling Repeater Jamming (ISRJ)

** Real UNKNOWN CLASSES  (K-real and U-real)
 - k-real: known-class real data (if available)
 - u-real: unknown real interference


Clean vs Impaired Semantics

* Clean Signals
 - Deterministic, ideal, noise-free
 - Used only for:
   ~ impairment reference
   ~ sanity checks
 - Clean signals do not cross the MATLAB → Python boundary
 - Clean signals are never used for training

* Impaired Signals
 - Generated as:
     x_imp = x_clean + n
 - Labels are inherited from clean signals
 - Impaired signals are the only signals exported

* SNR Policy
 - Training SNR range: [-5 dB, 20 dB]
 - Evaluation SNR range: [-10 dB, 20 dB]
 - SNR is explicitly recorded in metadata


Dataset Usage Rules (OPEN-SET)

- Training set:
  ~ k-mat only
- Validation set:
  ~ k-mat + u-mat
- Test / inference:
  ~ k-mat + u-mat + u-real
- No real data is ever used for training or threshold tuning


Randomness & Reproducibility

- Global random seed is fixed per dataset version
- No hidden randomness
- All stochastic components must be seed-controlled
- Regeneration with the same version must produce identical datasets


Data Integrity Expectations

- MATLAB d-type: float64
- Python d-type: float32
- Raw sample (single signal) shape in Python loader: (N,)
- Dataset raw tensor shape: (N_samples, N)
- Shape (STFT): (T, F) fixed by parameters
- Forbidden:
   ~ NaNs
   ~ Infs
   ~ Saturation
   ~ Silent clipping
- Python must validate, not assume.


Versioning Rule

* Any change to the following, requires a new specification version.
 - sampling
 - STFT parameters
 - normalization
 - class definitions
 - usage rules




