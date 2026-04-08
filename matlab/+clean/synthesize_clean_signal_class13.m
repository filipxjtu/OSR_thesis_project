function x_clean = synthesize_clean_signal_class13(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS13
% Triangular FM (TFM) - Unknown Class 4
% Explicit note about coupling:
    % - Acts as a geometric boundary case between LFM (Class 2) and SFM (Class 3).
    % - Uses discrete numerical integration to maintain precise phase continuity 
    %   across the piecewise linear frequency segments.

    N  = double(spec.N);
    fs = double(spec.fs);
    
    A       = params.A;
    delta_f = params.delta_f;
    fm      = params.fm;
    
    t = (0:N-1)' / fs;
    
    % creates a symmetric triangle wave bounded in [-1,1]
    f_inst = (delta_f / 2) * sawtooth(2 * pi * fm * t, 0.5);
    
    % Phase Integration
    phi_inst = (2 * pi / fs) * cumsum(f_inst);
    
    % Modulate
    x = exp(1i * phi_inst);
    
    % Apply Amplitude
    x = A * x;
    
    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'TFM: RMS is zero before normalization.');
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