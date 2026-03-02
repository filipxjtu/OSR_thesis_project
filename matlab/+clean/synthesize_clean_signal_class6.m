function x_clean = synthesize_clean_signal_class6(params, spec)
    % SYNTHESIZE_CLEAN_SIGNAL_CLASS6 Pure synthesis for FH signal (Class 6).
    % Frequency Hopping clean signal
    % x(t) = sum_{i=1}^I A * cos(2 * pi * f_i * t_i + phi_i); 

    % Initialize output vector
    x_clean = zeros(spec.N, 1);
    
    % Global time base
    t = double(0:spec.N-1)' / double(spec.fs);
    
    A      = params.A;
    Lhop   = params.hop_info.Lhop;
    phi_hops   = params.hop_info.phi_hops;
    hop_set  = params.hop_info.hop_set;
    n_hops = params.hop_info.n_hops;
    hop_idx  = params.hop_info.hop_idx;
    
    for i = 1:n_hops
        % get indicies for this segment
        idx0 = (i-1) * Lhop + 1;
        idx1   = min(i * Lhop, spec.N);
        
        %  frequency, phase and time for this segment
        f_i = hop_set(hop_idx(i));
        t_i = t(idx0:idx1);
        phi_i = phi_hops(i);
        
        % generate segment
        x_clean(idx0:idx1) = A * cos(2 * pi * f_i * t_i + phi_i);
    end
    
    % assertions
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(isreal(x_clean), 'Signal must be real-valued.');
    assert(all(isfinite(x_clean)), 'Signal contains Inf or NaN values.');
end