function x_clean = synthesize_clean_signal_class9(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS9
% Digital False Target Jamming (DFTJ)
% Explicit note about coupling:
    % - false targets are delayed/Doppler-shifted
    % - replicas of the actual intercepted LFM (Class 2) using same sample_index.
    % - Models physical DRFM behavior where jammer output depends deterministically on input.

    N  = double(spec.N);
    fs = double(spec.fs);

    L = params.dftj_info.L;
    Q = params.dftj_info.Q;

    t = (0:N-1)' / fs;

    % generate LFM target
    victim_sample_idx = clean.derive_victim_idx(params.sample_index, params.class_id);
    target_params = clean.generate_sample_params(2, victim_sample_idx, spec);
    s_target = clean.synthesize_clean_signal_class2(target_params, spec);

    % force RMS = 0.5
    rms_target = sqrt(mean(abs(s_target).^2));
    s_target = 0.5 * s_target / rms_target;

    % quantization
    I_clip = max(-1, min(1, real(s_target)));
    Q_clip = max(-1, min(1, imag(s_target)));

    s_mem = round(I_clip * L) / L + 1i * round(Q_clip * L) / L;

    % initialize output
    x = complex(zeros(N,1));

    % replica synthesis
    for q = 1:Q

        tau = params.dftj_info.tau(q);
        delta_f = params.dftj_info.delta_f(q);
        A_q = params.dftj_info.A_q(q);

        % true delay (zero-padding shift)
        shifted = zeros(N,1);
        if tau < N
            shifted((tau+1):end) = s_mem(1:(end-tau));
        end

        % Doppler
        doppler = exp(1i * 2*pi * delta_f * t);
        replica = A_q * shifted .* doppler;

        % Accumulate
        x = x + replica;
    end

    % amplitude
    x = params.A * x;

    % Normalize
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'DFTJ: RMS is zero.');
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