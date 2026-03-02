function x_clean = synthesize_clean_signal_class1(params, spec)
    % SYNTHESIZE_CLEAN_SIGNAL_CLASS1 Pure synthesis for Multi-Tone (Class 1).
    % Multi-Tone clean signal
    % x(t) = sum_{k=1}^K A_k * cos(2 * pi * f_k * t + phi_k)

    % calculate time base
    t = (double(0:spec.N-1)') / double(spec.fs);

    % initialize signal vector
    x_clean = zeros(spec.N, 1);

    % accumulate Tones
    for k = 1:params.K

        A_k  = params.A(k);
        f_k  = params.f0(k);
        phi_k = params.phi(k);
        
        % Add the tone to the total signal
        x_clean = x_clean + A_k * cos(2 * pi * f_k * t + phi_k);
    end

    % assertions
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(isreal(x_clean), 'Signal must be real-valued.');
    assert(all(isfinite(x_clean)), 'Signal contains Inf or NaN values.');
end