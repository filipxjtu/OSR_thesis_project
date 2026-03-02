function x_clean = synthesize_clean_signal_class5(params, spec)
    % SYNTHESIZE_CLEAN_SIGNAL_CLASS5 Generates a Noise FM (Class 5)
     % x(t) = A * cos( 2*pi*fc*t + 2*pi*kappa*integral + phi0 )

    A = params.A;
    nfft = params.bin_info.nfft;
    mod_bin_L = params.bin_info.bin_L;
    mod_bin_H = params.bin_info.bin_H;
    mod_phi_bins = params.bin_info.phi_bins;
    fc = params.fc;
    kappa = params.kappa;
    phi0 = params.phi;

    assert(nfft == spec.N, 'NFM: nfft must equal spec.N for v1.');

    % build the modulator signal m(t)
    M_spectrum = zeros(nfft, 1);
    M_spectrum(mod_bin_L : mod_bin_H) = exp(1i * mod_phi_bins);
    
    % enforce conjugate symmetry for a real modulator
    mirror_idx = nfft - (mod_bin_L : mod_bin_H) + 2;
    M_spectrum(mirror_idx) = conj(M_spectrum(mod_bin_L : mod_bin_H));
    
    % IFFT to get m(t)
    m_raw = ifft(M_spectrum, 'symmetric');
    
    % Normalize m(t) to have unit standard deviation (RMS)
    ss = std(m_raw);
    assert(ss > 0, 'NFM: std(m_raw) is zero; invalid spectrum construction.');
    m_t = m_raw / ss;

    % enforce zero-mean modulator (important for FM stability)
    m_t = m_t - mean(m_t);
    
    % instantaneous frequency Nyquist safety check
    f_inst_max = fc + kappa * max(abs(m_t));
    f_inst_min = fc - kappa * max(abs(m_t));

    assert(f_inst_max < spec.fs/2, 'NFM: instantaneous frequency exceeds Nyquist.');
    assert(f_inst_min > 0, 'NFM: instantaneous frequency below 0 Hz.');

    % setup Time and Carrier
    t = (double(0:spec.N-1)') / double(spec.fs);
    dt = 1 / double(spec.fs);
    
    % the integral part - integral of m(tau) from 0 to t
    m_integral = cumsum(m_t) * dt;
    
    % combine for final FM Equation
    phase = (2 * pi * fc * t) + (2 * pi * kappa * m_integral) + phi0;
    x_clean = A * cos(phase);
    
     % assertions
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(isreal(x_clean), 'Signal must be real-valued.');
    assert(all(isfinite(x_clean)), 'Signal contains Inf or NaN values.');
end