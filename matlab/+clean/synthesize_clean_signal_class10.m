function x_clean = synthesize_clean_signal_class10(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS10
% Moving-Band Noise (MBN) - Unknown Class 1
% Simulates a sweeping barrage jammer targeting multiple sub-bands sequentially.
% Uses an Overlap-Add (OLA) time-varying filter approach to ensure phase continuity.

    N  = double(spec.N);
    fs = double(spec.fs);
    
    A = params.A;
    B = params.mbn_info.B;
    delta_f_sweep = params.mbn_info.delta_f_sweep;
    f_start = params.mbn_info.f_start;
    L = double(params.mbn_info.frame_len);
    H = double(params.mbn_info.hop_len);
    
    % Generate base complex white Gaussian noise
    n_wgn = (randn(N, 1) + 1i * randn(N, 1)) / sqrt(2);
    
    % Initialize output signal
    x = complex(zeros(N, 1));
    
    % OLA Setup
    w = hann(L, 'periodic');
    f_vec = (0:L-1)' * (fs / L);
    
    % Shift frequency vector to match FFT layout (-fs/2 to fs/2 equivalent)
    f_vec(f_vec >= fs/2) = f_vec(f_vec >= fs/2) - fs;
    
    T_obs = N / fs;
    num_frames = floor((N - L) / H) + 1;
    
    for m = 1:num_frames
        idx_start = (m - 1) * H + 1;
        idx_end = idx_start + L - 1;
        
        % Extract noise segment
        n_seg = n_wgn(idx_start:idx_end);
        
        % Evaluate center frequency at the center of the current frame
        t_c = (idx_start + L/2) / fs;
        f_c = f_start + delta_f_sweep * (t_c / T_obs);
        
        % Transform to frequency domain
        N_seg = fft(n_seg);
        
        % Create ideal bandpass mask
        mask = (abs(f_vec - f_c) <= B/2);
        
        % Apply mask
        N_filt = N_seg .* mask;
        
        % Transform back to time domain
        n_filt_time = ifft(N_filt);
        
        % Apply Hann window for smooth overlap-add
        y_seg = n_filt_time .* w;
        
        % Accumulate
        x(idx_start:idx_end) = x(idx_start:idx_end) + y_seg;
    end
    
    % Apply amplitude
    x = A * x;
    
    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'MBN: RMS is zero before normalization.');
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