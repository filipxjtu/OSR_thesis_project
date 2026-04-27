function x_clean = synthesize_clean_signal_class14(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS12
% Range-Velocity Gate Pull-Off (RVGPO) - Unknown Class 3
% Explicit note about coupling:
    % - Victim signal is an LFM (Class 2) using the same sample_index.
    % - Models physical DRFM behavior where the ghost target dynamically walks away.

    N  = double(spec.N);
    fs = double(spec.fs);
    
    alpha = params.rvgpo_info.alpha;
    beta  = params.rvgpo_info.beta;
    L     = params.rvgpo_info.L;
    A_ghost = params.rvgpo_info.A_ghost;
    
    t = (0:N-1)' / fs;
    
    % Generate LFM target (Victim "Skin Return")
    target_params = clean.generate_sample_params(2, params.sample_index, spec);
    s_target = clean.synthesize_clean_signal_class2(target_params, spec);
    
    % Force RMS = 0.5 prior to DRFM quantization
    rms_target = sqrt(mean(abs(s_target).^2));
    assert(rms_target > 0, 'RVGPO: target RMS is zero.');
    s_target = 0.5 * s_target / rms_target;
    
    % DRFM Quantization
    I_clip = max(-1, min(1, real(s_target)));
    Q_clip = max(-1, min(1, imag(s_target)));
    s_mem = round(I_clip * L) / L + 1i * round(Q_clip * L) / L;
    
    % Apply Dynamic Walk-Off
    x_pull = complex(zeros(N, 1));
    
    % Precompute dynamic shift vectors
    tau_n = round(alpha * (t.^2) * fs);         % Instantaneous delay in samples
    phi_dop = 2 * pi * (0.5 * beta * (t.^2));   % Instantaneous Doppler phase
    
    for n = 1:N
        src_idx = n - tau_n(n);
        if src_idx >= 1 && src_idx <= N
            x_pull(n) = s_mem(src_idx) * exp(1i * phi_dop(n));
        end
    end
    
    % Combine the legitimate skin return with the deceptive pull-off ghost
    x = s_target + A_ghost * x_pull;
    
    % Apply amplitude
    x = params.A * x;
    
    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'RVGPO: RMS is zero before normalization.');
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