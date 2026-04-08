function x_clean = synthesize_clean_signal_class1(params, spec)
    % SYNTHESIZE_CLEAN_SIGNAL_CLASS1 
    % Multi-Tone Jamming (MTJ) with:
    % - multiple primary tones
    % - cubic PA nonlinearity
    % - short FIR memory filter
    %
    % Model:
    % s[n]    = sum_k A_k * exp(j*(2*pi*f_k*n/fs + phi_k))
    % x_nl[n] = s[n] + alpha * |s[n]|^2 * s[n]
    % x[n]    = h[n] * x_nl[n]

    % derive sample index
    n  = (0:double(spec.N)-1)';
    fs = double(spec.fs);

    K       = params.K;
    A_vec   = params.A;
    f_vec   = params.f0;
    phi_vec = params.phi;
    alpha   = params.alpha;
    tau_ns  = params.tau_ns;


    % initialize signal vector
    x_linear = complex(zeros(spec.N, 1));

    % accumulate Tones
    for k = 1:K
        
        % Add the tone to the total signal
        phase = (2 * pi * f_vec(k) * (n/fs) + phi_vec(k));
        x_linear = x_linear + A_vec(k) * exp(1i * phase);
    end

    % Apply third-order PA nonlinearity
    x_nonlinear = x_linear + alpha * (abs(x_linear).^2) .* x_linear;

    % Short FIR memory filter
    tau_sec = tau_ns * 1e-9;     % converted to seconds
    Ts = 1/fs;      % sampling period

    % 3-tap to 10-tap depending on tau
    filter_length = max(3, min(10, round(3 * tau_sec / Ts)));

    % create exponential decay impulse response
    n_tap = (0:filter_length-1)';
    h = exp(-n_tap * Ts / tau_sec);

    % normalize to preserve power
    h = h / sum(h);

    % the clean signal
    x_clean = filter(h, 1, x_nonlinear);

    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x_clean).^2));
    assert(rms_val > 0, 'MTJ: RMS is zero before normalization.');
    x_clean = x_clean / rms_val;

    % asserting
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length must match spec.N.');
    assert(~isreal(x_clean), 'Clean signal must be complex-valued.');
    assert(~all(imag(x_clean(:)) == 0), ...
        'Clean signal must have non-zero imaginary component.');
    assert(all(isfinite(x_clean(:))), 'Signal contains Inf or NaN values.');
end