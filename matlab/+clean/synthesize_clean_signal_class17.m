function x_clean = synthesize_clean_signal_class17(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS17
% AM-Tone Jamming (AMTJ) – Unknown Class 8 (FANET spoofing-flavored)
%
% A carrier amplitude-modulated by a low-rate sinusoid. Models simple
% spoofing/decoy approaches used against FANET command-and-control links
% where an adversary radiates a coarse imitation of a legitimate radio
% signal. Spectrally narrow with sidebands at fc ± fm — a useful "hard
% negative" because it carries spectral fine structure (unlike CWJ) but
% no chirp, hop, or wideband content (unlike LFMJ/FHJ/OFDMJ).
%
% Model:
%   m[n] = 1 + mod_index * cos(2*pi*fm*n/fs + phi_m)
%   x[n] = A * m[n] * exp(j*(2*pi*fc*n/fs + phi))

    N  = double(spec.N);
    fs = double(spec.fs);

    A         = params.A;
    fc        = params.fc;
    fm        = params.fm;
    mod_index = params.mod_index;
    phi       = params.phi;
    phi_m1     = params.phi_m1;

    t = (0:N-1)' / fs;

    % AM envelope (kept positive for proper AM; mod_index in [0.3, 0.9])
    m = 1 + mod_index * cos(2 * pi * fm * t + phi_m1);

    % Carrier
    c = exp(1i * (2 * pi * fc * t + phi));

    x = A * m .* c;

    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'AMTJ: RMS is zero before normalization.');
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