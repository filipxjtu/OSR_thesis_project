function x_clean = synthesize_clean_signal_class5(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS5
% Frequency Hopping Jamming with:
% - discrete frequency grid
% - PLL transient (amplitude suppression)
% - phase reset per hop

    N  = double(spec.N);
    fs = double(spec.fs);

    hop_info = params.hop_info;

    x = complex(zeros(N,1));

    idx_start = 1;

    for h = 1:hop_info.H

        Lh = hop_info.Lh(h);
        idx_end = idx_start + Lh - 1;

        n_local = (0:Lh-1)';
        t_local = n_local / fs;

        f_h = hop_info.f_grid(hop_info.hop_idx(h));
        phi_h = hop_info.phi_h(h);

        % Base tone
        s = exp(1i * (2*pi*f_h*t_local + phi_h));

        % PLL transient window
        N_trans = hop_info.N_trans(h);

        w = ones(Lh,1);
        if N_trans > 1
            n_ramp = (0:N_trans-1)';
            
             % amplitude ramp (half-Hamming(Hamm: 0.54 - 0.46*cos(2*pi*n/(N-1))
            amp_ramp = 0.54 - 0.46 * cos(pi * n_ramp / (N_trans - 1));
            w(1:N_trans) = amp_ramp; % store in window

            % phase settling
            tau = N_trans/3;   % time constant (settle by end of transiet)
            phase_error = ((2*rand()-1) * 0.3) * exp(-n_ramp / tau); % bounded random intial offset
            phase_settle = exp(1i * phase_error);

            % apply both to tone segment
            s(1:N_trans) = phase_settle .* s(1:N_trans);
        end

        s_glitch = w .* s;

        % Insert into signal
        x(idx_start:idx_end) = s_glitch;

        idx_start = idx_end + 1;
    end

    % amplitude
    x = params.A * x;

    % Normalize
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'FHJ: RMS is zero before normalization.');
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