function x_clean = synthesize_clean_signal_class12(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS12
% Impulsive α-Stable Noise Jammer
%
% Heavy-tailed noise jammer. With RMS normalization preserved for pipeline
% consistency, the discriminative axis is the stability index alpha (lower
% alpha => heavier tails => higher kurtosis surviving the RX chain).
%
% Model:
%   x[n] = alpha_stable_complex(alpha, beta, gamma, delta)
%   then unit-RMS normalized.

    N  = double(spec.N);

    A     = params.A;
    alpha = params.alpha;
    beta  = params.beta;
    gamma = params.gamma;
    delta = params.delta;

    % Guard against numerical instability near alpha = 1
    assert(abs(alpha - 1) > 0.05, ...
        'alpha-stable: alpha too close to 1, CMS sampler unstable.')

    % Generate real and imaginary parts as independent α‑stable
    % Using CMS method (see Chambers, Mallows, Stuck 1976)
    x_real = alpha_stable_rvs(N, alpha, beta, gamma, delta);
    x_imag = alpha_stable_rvs(N, alpha, beta, gamma, delta);
    x = A * (x_real + 1i * x_imag);

    mad_x = median(abs(x - median(x)));
    if mad_x > 0
        clip_thresh = 50 * mad_x;   % keeps the heavy tail, kills pathologies
        mag = abs(x);
        scale = min(1, clip_thresh ./ (mag + eps));
        x = x .* scale;
    end

    % Normalise to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    if rms_val == 0
        rms_val = eps;
    end
    x_clean = x / rms_val;

    % Assertions
    assert(iscolumn(x_clean), 'Output must be column.');
    assert(numel(x_clean) == spec.N, 'Length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex.');
    assert(~all(imag(x_clean(:)) == 0), 'Imag part must exist.');
    assert(all(isfinite(x_clean(:))), 'Inf/NaN found.');
end

function z = alpha_stable_rvs(N, alpha, beta, gamma, delta)
% Chambers-Mallows-Stuck sampler for univariate alpha-stable variates.
    U = pi * (rand(N,1) - 0.5);
    W = -log(rand(N,1));
    if beta == 0
        z = gamma * sin(alpha * U) ./ (cos(U).^(1/alpha)) .* ...
            (cos((1-alpha)*U) ./ W).^((1-alpha)/alpha) + delta;
    else
        t = tan(pi*alpha/2);
        B = atan(beta * t) / alpha;
        S = (1 + (beta * t).^2).^(1/(2*alpha));
        z = gamma * S * sin(alpha*(U + B)) ./ (cos(U).^(1/alpha)) .* ...
            (cos((1-alpha)*U - alpha*B) ./ W).^((1-alpha)/alpha) + delta;
    end
end