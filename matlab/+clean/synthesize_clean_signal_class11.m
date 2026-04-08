function x_clean = synthesize_clean_signal_class11(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS11
% Intermittent OFDM (I-OFDM) - Unknown Class 2
% Explicit note about coupling:
    % - uses the exact Class 6 (OFDM) generator for the base signal using same sample_index.
    % - applies a randomized smoothed square-wave mask for burstiness.

    N = double(spec.N);
    
    % Generate continuous OFDM base signal
    target_params = clean.generate_sample_params(6, params.sample_index, spec);
    s_ofdm = clean.synthesize_clean_signal_class6(target_params, spec);
    
    % Retrieve burst constraints
    D_b_range = params.iofdm_info.D_b_range;
    G_b_range = params.iofdm_info.G_b_range;
    taps = params.iofdm_info.taper_taps;
    
    % Generate burst mask
    m_raw = zeros(N, 1);
    idx = randi([1, 100]); % Random start offset
    
    while idx <= N
        % Draw burst and gap durations for this cycle
        D_b = randi(D_b_range);
        G_b = randi(G_b_range);
        
        idx_end = min(idx + D_b - 1, N);
        m_raw(idx:idx_end) = 1;
        
        % Move index to the start of the next burst
        idx = idx + D_b + G_b;
    end
    
    % Apply hardware tapering (smoothing) to the mask
    h_taper = ones(taps, 1) / taps;
    m_smooth = filter(h_taper, 1, m_raw);
    
    % Modulate base signal
    x = s_ofdm .* m_smooth;
    
    % Apply Amplitude
    x = params.A * x;
    
    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'I-OFDM: RMS is zero before normalization.');
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