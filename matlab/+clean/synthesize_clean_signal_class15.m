function x_clean = synthesize_clean_signal_class15(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS15
% Slow Sweep Jamming (SWPJ) – Unknown Class 6 (FANET anti-FH)
%
% A linearly-swept tone with a SLOW sweep period, designed to "smear"
% interference across a frequency-hopping pattern. Distinct from LFMJ
% (Class 2) by sweep rate: LFMJ sweeps over the observation window with a
% rate set by chirp_rate, while SWPJ uses a much smaller sweep excursion
% per unit time and resets via sawtooth, producing repeated slow ramps.
%
% Model:
%   f_inst[n] = f_start + delta_f_sweep * frac(t / T_sweep)
%   phi[n]    = 2*pi/fs * cumsum(f_inst)
%   x[n]      = A * exp(j * phi[n])

    N  = double(spec.N);
    fs = double(spec.fs);

    A             = params.A;
    f_start       = params.f0;
    delta_f_sweep = params.delta_f;
    T_sweep       = params.T;          % sweep period in seconds
    phi0          = params.phi;

    t = (0:N-1)' / fs;

    % Sawtooth in [0, 1) with period T_sweep
    ramp = mod(t, T_sweep) / T_sweep;

    % Instantaneous frequency
    f_inst = f_start + delta_f_sweep * ramp;

    % Phase via numerical integration
    phase = 2 * pi * cumsum(f_inst) / fs + phi0;

    x = A * exp(1i * phase);

    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'SWPJ: RMS is zero before normalization.');
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