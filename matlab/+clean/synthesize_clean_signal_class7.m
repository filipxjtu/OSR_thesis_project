function x_clean = synthesize_clean_signal_class7(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS7
% Periodic Gaussian Pulse Jamming with:
% - pulse train structure
% - timing jitter
% - pulse-to-pulse amplitude variation

    N  = double(spec.N);
    fs = double(spec.fs);

    n = (0:N-1)';
    t = n / fs;

    % build Gaussian pulse envelope
    p = zeros(N, 1);

    sigma = double(params.sigma);
    assert(sigma > 0, 'PGPJ: sigma must be positive.');

    for m = 1:params.pulse_info.M
        c_m = params.pulse_info.centers(m);
        eta_m = params.pulse_info.eta(m);

        p = p + eta_m * exp(-((n - c_m).^2) / (2 * sigma^2));
    end

    % apply carrier
    carrier = exp(1i * (2*pi*params.fc*t + params.phi));
    x = params.A * p .* carrier;

    % normalize
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'PGPJ: RMS is zero before normalization.');
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