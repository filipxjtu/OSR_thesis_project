function x_clean = synthesize_clean_signal_class2(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS2
% LFM with:
% - Doppler time scaling
% - Tukey window (hardware pulse shaping)

    N = spec.N;    
    A = params.A;
    f0 = params.f0;
    T = params.T;
    K = params.K;
    phi = params.phi;
    alpha = params.alpha_tukey;

    % Time base
    t = (double(0:spec.N-1)') / double(spec.fs);

    assert(T > 0, 'Sweep duration T must be positive.');

    % Doppler scaling
    t_s = (1 + params.delta) * t;

    % Phase
    phase = 2 * pi * (f0 * t_s + 0.5 * K * (t_s.^2)) + phi;
    x = A * exp(1i * phase);

    % Tukey window
    w = tukeywin(N, alpha);
 
    x = x .* w;

    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'LFM: RMS is zero before normalization.');
    x_clean = x / rms_val;

    % Assertions
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex.');
    assert(~all(imag(x_clean(:)) == 0), 'Imaginary part must exist.');
    assert(all(isfinite(x_clean(:))), 'Signal contains NaN/Inf.');
end