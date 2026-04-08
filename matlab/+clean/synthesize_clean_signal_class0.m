function x_clean = synthesize_clean_signal_class0(params, spec)
    % SYNTHESIZE_CLEAN_SIGNAL_CLASS0

    % Generates a complex base band Single Tone clean signal with
    %  - a constant frequency offset (delta_f)
    %  - Wiener phase noise with reflection bound
    % phi[n] = 2 * pi * (f0 + delta_f)* (n/fs) + phi + theta[n]
    % x[n] = A * exp(j * Phi[n])

    % derive sample index
    n = (0:double(spec.N)-1)';
    fs = double(spec.fs);
    t = n/fs;

    A = params.A;
    f0 = params.f0;
    delta_f = params.delta_f;
    phi = params.phi;
    sigma = params.sigma;

    % Wiener phase noise
    theta = zeros(spec.N, 1);
    for k = 2:spec.N
        w_k = sigma * randn(1) * sqrt(1/fs);    % Wiener increment
        theta(k) = theta(k-1) + w_k;

        % wrap to [-pi, +pi]
        theta(k) = mod(theta(k) + pi, 2*pi) - pi;
    end

    % total instantaneous phase
    phi_freq = 2*pi * ((f0 + delta_f) .* t);
    Phi_inst = phi_freq + phi + theta;

    % generate complex baseband
    x_clean = A * exp(1i * Phi_inst);

    % normalize to unit RMS
    rms_val = sqrt(mean(abs(x_clean).^2));
    assert(rms_val > 0, 'STJ: RMS is zero before normalization.');
    x_clean = x_clean / rms_val;

    % asserting
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length must match spec.N.');
    assert(~isreal(x_clean), 'Clean signal must be complex-valued.');
    assert(~all(imag(x_clean(:)) == 0), ...
        'Clean signal must have non-zero imaginary component.');
    assert(all(isfinite(x_clean(:))), 'Signal contains Inf or NaN values.');
end