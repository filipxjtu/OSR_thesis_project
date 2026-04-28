function x_clean = synthesize_clean_signal_class8(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS8
% Sliced-Repeating Jamming / ISRJ with:
% - dynamic target selection (LFM or OFDM)
    % Explicit note about coupling:
    % - intentionally reuses Class 2 (LFM) and Class 6 (OFDM) generators
    % - target waveform uses same sample_index
    % - models DRFM behavior where jammer processes the actual intercepted signal.
% - independent I/Q clipping and low-bit quantization
% - pulse-to-pulse amplitude drift
% - discrete timing jitter
% - additive overlap

    N = double(spec.N);
    K = double(params.K);
    L = params.srj_info.L;
    D = double(params.srj_info.D);
    M = double(params.srj_info.M);
    T_pri = double(params.srj_info.T_pri);
    target_type = params.srj_info.target_type;


    % generate target waveform using existing class logic
    victim_sample_idx = clean.derive_victim_idx(params.sample_index, params.class_id);

    if strcmp(target_type, "lfm")
        target_params = clean.generate_sample_params(2, victim_sample_idx, spec);
        s_target = clean.synthesize_clean_signal_class2(target_params, spec);
    elseif strcmp(target_type, "ofdm")
        target_params = clean.generate_sample_params(6, victim_sample_idx, spec);
        s_target = clean.synthesize_clean_signal_class6(target_params, spec);
    else
        error('ISRJ: invalid target_type.');
    end

    % force RMS to 0.5 so natural clipping occurs
    rms_target = sqrt(mean(abs(s_target).^2));
    assert(rms_target > 0, 'ISRJ: target RMS is zero.');
    s_target = 0.5 * s_target / rms_target;

    % independent I/Q clipping and quantization
    I_clip = max(-1, min(1, real(s_target)));
    Q_clip = max(-1, min(1, imag(s_target)));

    s_mem = round(I_clip * L) / L + 1i * round(Q_clip * L) / L;

    % initialize output
    x = complex(zeros(N,1));

    % slice and repeat loop
    for m = 0:(M-1)
        idx = m * T_pri + 1;

        % boundary guard for extraction
        if (idx + D - 1) > N
            break;
        end

        slice = s_mem(idx : idx + D - 1);
        gamma_m = params.srj_info.gamma(m + 1);

        for k = 1:K
            epsilon_k = params.srj_info.epsilon(m + 1, k);
            paste_idx = idx + k * D + epsilon_k;

            % boundary guard for paste
            if paste_idx < 1
                continue;
            end
            if (paste_idx + D - 1) > N
                continue;
            end

            if params.srj_info.use_additive_overlap
                x(paste_idx : paste_idx + D - 1) = ...
                    x(paste_idx : paste_idx + D - 1) + gamma_m * slice;
            else
                x(paste_idx : paste_idx + D - 1) = gamma_m * slice;
            end
        end
    end

    % amplitude
    x = params.A * x;

    % nnormalize
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'ISRJ: RMS is zero before normalization.');
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