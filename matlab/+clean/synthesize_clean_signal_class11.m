function x_clean = synthesize_clean_signal_class11(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS15
% Chaotic Carrier Jammer (logistic-map driven phase)
%
% Model:
%   x[n] = A * exp(j * (2*pi*fc*t[n] + phi[n]))
%   phi[n] = 2*pi * scale * y[n]
%   y[n] = r * y[n-1] * (1 - y[n-1])   (logistic map)
%
% The instantaneous phase is driven by a deterministic chaotic sequence.
% The resulting signal has Gaussian-like second-order statistics but a
% strongly non-zero bispectrum (unique among your 0-9 classes).
%
% Parameters:
%   A         : amplitude (linear)
%   fc        : carrier frequency (Hz)
%   phi0      : initial phase (rad)
%   r         : logistic map parameter (typically 3.9 to 4.0) – chaos
%   scale     : scaling factor to spread phase over [0, 2π) (e.g., 1.0)
%   seed_y    : initial y(0) in (0,1), e.g., 0.3 (randomised)
%
% Output:
%   x_clean   : column vector, complex, unit RMS, length spec.N

    N  = double(spec.N);
    fs = double(spec.fs);
    t  = (0:N-1)' / fs;

    A     = params.A;
    fc    = params.fc;
    phi0  = params.phi;
    Rc     = params.Rc;
    sigma = params.sigma;
    alpha   = params.alpha;

    % Generate logistic map sequence
    y = zeros(N, 1);
    y(1) = alpha;
    for n = 2:N
        y(n) = Rc * y(n-1) * (1 - y(n-1));
    end

    % Phase = chaotic value scaled and wrapped
    phi_chaotic = 2 * pi * sigma * y;
    % Optional: add small diffusion to avoid periodicity? Not needed for chaos.

    % Total instantaneous phase
    phase = 2 * pi * fc * t + phi_chaotic + phi0;

    x = A * exp(1i * phase);

    % Normalise to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'Chaotic: RMS zero');
    x_clean = x / rms_val;

    % Assertions
    assert(iscolumn(x_clean), 'Output must be column.');
    assert(numel(x_clean) == spec.N, 'Length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex.');
    assert(~all(imag(x_clean(:)) == 0), 'Imag part must exist.');
    assert(all(isfinite(x_clean(:))), 'Inf/NaN found.');
end