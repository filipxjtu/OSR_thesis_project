function x_clean = synthesize_clean_signal_class3(params, spec)
    % SYNTHESIZE_CLEAN_SIGNAL_CLASS3
    % Sinusoidal FM clean signal with:
    % - primary modulation harmonic
    % - secondary harmonic distortion
    %
    % Model:
    %   prim_harm = beta1 * sin(2*pi*fm*t + phi_m1)
    %   secd_harm = beta2 * sin(4*pi*fm*t + phi_m2)
    %   Phi(t) = 2*pi*fc*t + prim_harm + secd_harm + phi
    %   x(t) = A * exp(j * Phi(t))

    A     = params.A;
    fc    = params.fc;
    fm    = params.fm;
    phi   = params.phi;
    phi_m1 = params.phi_m1;
    phi_m2 = params.phi_m2;
    beta  = params.beta;
    beta2  = params.beta2;

    % Time base
    t = (double(0:spec.N-1)') / double(spec.fs);
  
    % Harmonic phase components
    prim_harm = beta * sin(2 * pi * fm * t + phi_m1);
    secd_harm = beta2 * sin(4 * pi * fm * t + phi_m2);

    % Total phase
    phase = 2 * pi * fc * t + prim_harm + secd_harm + phi;

    % Complex baseband signal
    x = A * exp(1i * phase);

    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'SFM: RMS is zero before normalization.');
    x_clean = x / rms_val;

    % Assertions
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex-valued.');
    assert(~all(imag(x_clean(:)) == 0), ...
        'Signal must have non-zero imaginary component.');
    assert(all(isfinite(x_clean(:))), ...
        'Signal contains Inf or NaN values.');
end