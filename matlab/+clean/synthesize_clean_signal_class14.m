function x_clean = synthesize_clean_signal_class14(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS14
% Continuous-Wave Jamming (CWJ) – Unknown Class 5 (FANET DoS)
%
% A single unmodulated carrier at fc with constant envelope. Canonical
% jamming threat for FANET and the simplest hardware to deploy (signal
% generator + amplifier). Used in the literature as the baseline DoS
% jammer against FH-CDMA and OFDM uplinks.
%
% Model:
%   x[n] = A * exp(j*(2*pi*fc*n/fs + phi))
%
% Spectrally a delta function at fc — distinct from every known class
% which all carry chirp / hop / modulation / noise structure.

    N  = double(spec.N);
    fs = double(spec.fs);

    A   = params.A;
    fc  = params.fc;
    phi = params.phi;

    n = (0:N-1)';
    t = n / fs;

    x = A * exp(1i * (2 * pi * fc * t + phi));

    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'CWJ: RMS is zero before normalization.');
    x_clean = x / rms_val;

    % Assertions
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex.');
    assert(~all(imag(x_clean(:)) == 0), ...
        'Signal must have non-zero imaginary component.');
    assert(all(isfinite(x_clean(:))), ...
        'Signal contains NaN/Inf.');
end