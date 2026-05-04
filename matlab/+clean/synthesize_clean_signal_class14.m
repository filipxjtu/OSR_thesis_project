function x_clean = synthesize_clean_signal_class14(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS13
% Triangular FM (TFM) - Robust Variant for Unknown Class 4
% Evolution from baseline:
%   - Replaced fixed 0.5 duty cycle with params.symmetry to model VCO bias.
%   - Introduced movmean smoothing on f_inst to model PLL transient response.

    N  = double(spec.N);
    fs = double(spec.fs);
    
    A       = params.A;
    delta_f = params.delta_f;
    fm      = params.fm;
    K     = params.K; % Hardware-realistic asymmetry
    
    t = (0:N-1)' / fs;
    
    % Generate raw triangular frequency trajectory
    f_inst = (delta_f / 2) * sawtooth(2 * pi * fm * t, K);
    
    % Tweak: Apply smoothing (5-sample window) to vertices
    % Simulates finite bandwidth of the jammer's frequency modulator
    f_inst = movmean(f_inst, 5);
    
    % Phase Integration
    phi_inst = (2 * pi / fs) * cumsum(f_inst);
    
    % Modulate and scale
    x = A * exp(1i * phi_inst);
    
    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'TFM: RMS is zero before normalization.');
    x_clean = x / rms_val;
    
    % Standard Validations
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex.');
    assert(all(isfinite(x_clean(:))), 'Signal contains NaN/Inf.');
end