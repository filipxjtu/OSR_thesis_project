function x_clean = synthesize_clean_signal_class16(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS 16
% Pulsed Gaussian Noise Jamming (PGNJ) – Unknown Class (FANET burst jammer)
%
% Bursty broadband-noise jammer used to evade energy detection by
% radiating only during short pulses. Same noise statistics as PBNJ
% (Class 4) but with a low duty cycle and abrupt on/off envelope.
% Teaches the calibrator to attend to time-domain envelope, not just
% spectral support.
%
% Model:
%   n_w[n]  = (randn + j*randn) / sqrt(2)             % complex WGN
%   g[n]    = pulse train at PRF with width = duty * T_pulse
%   x[n]    = A * (g[n] .* n_w[n])

    N  = double(spec.N);
    fs = double(spec.fs);

    A          = params.A;
    PRF        = params.pgn_info.PRF;          % pulse repetition frequency (Hz)
    duty_cycle = params.pgn_info.duty_cycle;   % fraction in (0, 1]
    rise_samp  = double(params.pgn_info.rise_samp);  % envelope rise/fall length (samples)

    % Period in samples
    T_per_samp = round(fs / PRF);
    on_samp    = round(duty_cycle * T_per_samp);
    assert(on_samp >= 2, 'PGNJ: on-time too short for chosen PRF/duty.');

    % Build pulse envelope with smooth rise/fall to avoid spectral splatter
    g = zeros(N, 1);
    rise_samp = min(rise_samp, floor(on_samp / 2));
    if rise_samp > 0
        ramp_up   = (1 - cos(pi * (0:rise_samp-1)' / rise_samp)) / 2;
        ramp_down = flipud(ramp_up);
    else
        ramp_up   = [];
        ramp_down = [];
    end
    flat_len = on_samp - 2 * rise_samp;
    pulse = [ramp_up; ones(max(flat_len, 0), 1); ramp_down];
    pulse_len = numel(pulse);

    idx = 1;
    while idx <= N
        idx_end = min(idx + pulse_len - 1, N);
        g(idx:idx_end) = pulse(1:(idx_end - idx + 1));
        idx = idx + T_per_samp;
    end

    % Complex white Gaussian noise
    n_w = (randn(N, 1) + 1i * randn(N, 1)) / sqrt(2);

    x = A * (g .* n_w);

    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'PGNJ: RMS is zero before normalization.');
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