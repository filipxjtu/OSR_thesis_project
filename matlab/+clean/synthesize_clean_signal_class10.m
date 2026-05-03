function x_clean = synthesize_clean_signal_class10(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS17
% Impulsive α‑Stable Noise Jammer
%
% Model:
%   x[n] = A * S_alpha(1, 0, 0)   (complex isotropic α‑stable)
%   where the real and imaginary parts are jointly α‑stable.
%   We use the Chambers–Mallows–Stuck method to generate independent
%   α‑stable variates and combine them into complex isotropic.
%
% Parameters:
%   A       : amplitude (linear)
%   alpha   : stability index (typically 1.4 for impulsive)
%   beta    : symmetry parameter (0 for symmetric)
%   gamma   : scale > 0
%   delta   : location (0 for zero‑mean)
%
% Output:
%   x_clean : column vector, complex, unit RMS

    N  = double(spec.N);
    fs = double(spec.fs);  % not used but kept for uniformity

    A     = params.A;
    alpha = params.alpha;
    beta  = params.beta;
    gamma = params.gamma;
    delta = params.delta;

    % Generate real and imaginary parts as independent α‑stable
    % Using CMS method (see Chambers, Mallows, Stuck 1976)
    x_real = alpha_stable_rvs(N, alpha, beta, gamma, delta);
    x_imag = alpha_stable_rvs(N, alpha, beta, gamma, delta);
    x = A * (x_real + 1i * x_imag);

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
% Generate N samples of symmetric (beta=0) α‑stable distribution.
% Chambers–Mallows–Stuck algorithm.
    U = pi * (rand(N,1) - 0.5);
    W = -log(rand(N,1));
    if beta == 0
        % Symmetric case: simpler formula
        z = gamma * sin(alpha * U) ./ (cos(U).^(1/alpha)) .* ...
        (cos((1-alpha)*U) ./ W).^((1-alpha)/alpha) + delta;
    else
        % General case (not needed for beta=0 but included for completeness)
        t = tan(pi*alpha/2);
        B = atan(beta * t) / alpha;
        S = (1 + (beta * t).^2).^(1/(2*alpha));
        z = gamma * S * sin(alpha*(U + B)) ./ (cos(U).^(1/alpha)) .* ...
            (cos((1-alpha)*U - alpha*B) ./ W).^((1-alpha)/alpha) + delta;
    end
end