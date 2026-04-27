function x_clean = synthesize_clean_signal_class12(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS13
% Direct-Sequence Spread Spectrum (DSSS) – Unknown Class
%
% Generates a BPSK‑modulated spreading waveform with:
%  - random binary chip sequence
%  - raised‑cosine pulse shaping (bandwidth control)
%  - carrier upconversion
%  - unit‑RMS normalisation

    N  = double(spec.N);
    fs = double(spec.fs);

    A    = params.A;
    beta = params.beta;
    Rc   = params.Rc;
    fc   = params.fc;
    phi  = params.phi;

    Tc = 1 / Rc;                     % chip duration (seconds)
    Nc = ceil((N/fs) / Tc) + 1;      % enough chips to cover the whole observation

    % --- random chip sequence (uses current RNG state) ---
    chips = 2 * (randi([0,1], Nc, 1)) - 1;   % +/- 1

    % --- raised‑cosine pulse shape (sampled at fs) ---
    span = 6;                                  % symbol periods each side
    t_filter = (-span*Tc : 1/fs : span*Tc)';   % time vector for impulse response
    % Avoid exactly zero to prevent division by zero
    tiny = 1e-12;
    t_filter(abs(t_filter) < tiny) = tiny;

    % Raised‑cosine formula (full, not root‑raised)
    num = sin(pi * t_filter / Tc) .* cos(pi * beta * t_filter / Tc);
    den = (pi * t_filter / Tc) .* (1 - (2 * beta * t_filter / Tc).^2);
    h_rc = num ./ den;

    % Correct the limit at t = 0
    idx0 = (abs(t_filter) < 1e-12);
    h_rc(idx0) = 1 - beta + 4*beta/pi;

    h_rc = h_rc / sum(h_rc);          % normalise to unit area
    Lh = length(h_rc);

    % --- build impulse train ---
    imp_train = zeros(N + Lh, 1);     % extra room for filter tail
    for k = 1:Nc
        t_k = (k-1) * Tc;
        n0 = round(t_k * fs) + 1;     % nearest sample index (1‑based)
        if n0 > 0 && n0 <= N + Lh
            imp_train(n0) = chips(k);
        end
    end

    % --- apply pulse‑shaping filter ---
    x_shaped = conv(imp_train, h_rc, 'same');   % 'same' keeps central N+Lh samples
    % Keep only the valid N samples (discard filter transients)
    x_base = x_shaped(1:N);

    % --- upconvert to carrier ---
    t = (0:N-1)' / fs;
    x = A * x_base .* exp(1i * (2 * pi * fc * t + phi));

    % --- normalise to unit RMS ---
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'DSSS: RMS is zero before normalization.');
    x_clean = x / rms_val;

    % --- assertions ---
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex.');
    assert(~all(imag(x_clean(:)) == 0), ...
        'Signal must have non-zero imaginary component.');
    assert(all(isfinite(x_clean(:))), 'Signal contains NaN/Inf.');
end