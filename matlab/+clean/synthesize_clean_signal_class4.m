function x_clean = synthesize_clean_signal_class4(params, spec)
     % SYNTHESIZE_CLEAN_SIGNAL_CLASS4 Generates PBN signal (Class 4).
    % Partial Band Noise clean signal
    
    A = params.A;
    nfft = params.bin_info.nfft;
    bin_L = params.bin_info.bin_L;
    bin_H = params.bin_info.bin_H;
    phi_bins = params.bin_info.phi_bins;
    
    assert(nfft == spec.N, 'PBN: nfft must equal spec.N for v1.');

    % initialize complex spectrum (zero-filled)
    X = zeros(nfft, 1);
    
    % map stored phases to the positive frequency band
    X(bin_L : bin_H) = exp(1i * phi_bins);
     
    % enforce conjugate symmetry to ge real-valued time signal
    mirror_idx = nfft - (bin_L : bin_H) + 2;
    
    % apply the conjugate of the positive bins to the negative bins
    X(mirror_idx) = conj(X(bin_L : bin_H));
    
    % transform to time domain
    x_raw = ifft(X, 'symmetric'); 
    
    % normalization and scaling
    s = std(x_raw);
    assert(s > 0, 'PBN: std(x_raw) is zero; invalid spectrum construction.');
    x_norm = x_raw / s;
    
    x_clean = A * x_norm;
    
    % assertions
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(isreal(x_clean), 'Signal must be real-valued.');
    assert(all(isfinite(x_clean)), 'Signal contains Inf or NaN values.');
end