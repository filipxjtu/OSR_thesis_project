function x_clean = synthesize_clean_signal_class0(params, spec)
    % SYNTHESIZE_CLEAN_SIGNAL_CLASS0 Generates a pure cosine wave (Class 0).
    % Single Tone clean signal
    % x(t) = A * cos(2 * pi * f0 * t + phi)

    % derive time base from spec
    t = double(0:spec.N-1)' / double(spec.fs);

    % generate signal
    x_clean = params.A * cos(2 * pi * params.f0 * t + params.phi);

    % asserting
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length must match spec.N.');
    assert(isreal(x_clean), 'Clean signal must be real-valued.');
    assert(all(isfinite(x_clean)), 'Signal contains Inf or NaN values.');
end