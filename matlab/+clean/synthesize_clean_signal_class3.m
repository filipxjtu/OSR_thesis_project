function x_clean = synthesize_clean_signal_class3(params, spec)
    % SYNTHESIZE_CLEAN_SIGNAL_CLASS3 Pure synthesis for Multi-Tone (Class 3).
    % Sinusoidal Frequency Modulated clean signal 
    % x(t) = A * cos((2 * pi * fc * t) - (beta * cos (2 * pi * fm * t)) + phi),  
    % where beta = df / fm
    % instantaneous frequency is f(t) = fc + df * sin(2 * pi * fm * t)

    % calculate time base
    t = (double(0:spec.N-1)') / double(spec.fs);

    A  = params.A;
    fc  = params.fc;
    fm = params.fm;
    phi = params.phi;
    beta = params.beta;
  
    % generate signal
    phase = (2 * pi * fc * t) - (beta * cos (2 * pi * fm * t)) + phi;
    x_clean = A * cos(phase);
    
    % assertions
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(isreal(x_clean), 'Signal must be real-valued.');
    assert(all(isfinite(x_clean)), 'Signal contains Inf or NaN values.');
end