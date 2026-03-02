function x_clean = synthesize_clean_signal_class2(params, spec)
    % SYNTHESIZE_CLEAN_SIGNAL_CLASS2 Pure synthesis for LFM (Class 2).
    %Linear Frequency Modulated clean signal (chirp)
    % x(t) = A * cos(2 * pi ( f0 * t + 0.5 * k * t * t) + phi),  
    % where k = (f1 - f0) / T
    % instantaneous frequency is f(t) = f0 + kt

    % calculate time base
    t = (double(0:spec.N-1)') / double(spec.fs);

    % generate signal
    assert(params.T > 0, 'Sweep duration T must be positive.');
    k = (params.f1 - params.f0) / params.T;
    
    x_clean = params.A * cos( 2 * pi * (params.f0 * t + (0.5 * k * t .^ 2)) + params.phi);
    
    % assertions
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(isreal(x_clean), 'Signal must be real-valued.');
    assert(all(isfinite(x_clean)), 'Signal contains Inf or NaN values.');
end