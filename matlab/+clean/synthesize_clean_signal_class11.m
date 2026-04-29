function x_clean = synthesize_clean_signal_class11(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS11
% CP-2FSK (Continuous-Phase Binary FSK) – Unknown Class
%
% Emits a constant-envelope signal whose instantaneous frequency
% switches between fc - f_dev (bit 0) and fc + f_dev (bit 1).
% Phase is continuous across symbol boundaries.
% Uses random binary data with symbol rate Rs (Baud).

    N  = double(spec.N);
    fs = double(spec.fs);

    A     = params.A;
    Rc    = params.Rc;          % symbol rate
    delta_f = params.delta_f;       % frequency deviation
    fc    = params.fc;          % carrier
    phi0  = params.phi;         % initial phase

    Ts = 1 / Rc;
    Nsym = ceil(N / (Ts * fs)) + 2;   % enough symbols to cover N samples
    bits = randi([0,1], Nsym, 1);     % random data

    % Build time vector and symbol boundaries
    t = (0:N-1)' / fs;
    sym_boundaries = (0:Nsym) * Ts;
    % For each sample, find its symbol index
    sym_idx = sum(t >= sym_boundaries, 2);
    sym_idx = min(sym_idx, Nsym);

    % Instantaneous frequency per sample
    f_inst = fc + (2*double(bits(sym_idx)) - 1) * delta_f;

    % Continuous phase: integrate frequency
    phase = 2 * pi * cumsum(f_inst) / fs + phi0;

    x = A * exp(1i * phase);

    % Normalise to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'CPFSK: RMS is zero before normalization.');
    x_clean = x / rms_val;

    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex.');
    assert(~all(imag(x_clean(:)) == 0), ...
        'Signal must have non-zero imaginary component.');
    assert(all(isfinite(x_clean(:))), 'Signal contains NaN/Inf.');
end