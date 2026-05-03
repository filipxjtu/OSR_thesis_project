function x_clean = synthesize_clean_signal_class13(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS14
% Phase-Coded Pulse Jammer (Barker-13 or P4 polyphase)
%
% Model:
%   x(t) = A * rect(t / T_pulse) * exp(j*(2*pi*fc*t + phi_code(t) + phi0))
%   phi_code(t) is piecewise constant over chips of width T_chip = T_pulse / N_chips
%
% Parameters:
%   A           : amplitude (linear)
%   fc          : carrier frequency (Hz)
%   phi0        : initial phase (rad)
%   code_type   : 'barker13' or 'p4'
%   N_chips     : number of chips (e.g., 13 for Barker, up to 64 for P4)
%   T_pulse     : pulse duration (seconds)
%   PRF         : pulse repetition frequency (Hz) – interval between pulse starts
%   M           : number of pulses within the observation window
%   t_start     : start time of first pulse (seconds)
%
% Output:
%   x_clean     : column vector, complex, unit RMS, length spec.N

    N  = double(spec.N);
    fs = double(spec.fs);
    t  = (0:N-1)' / fs;

    A       = params.A;
    fc      = params.fc;
    phi0    = params.phi;
    code    = params.pulse_info.code_phase;        % vector of phases per chip (rad)
    N_chips = params.pulse_info.N_chips;
    T_pulse = params.pulse_info.T_pulse;
    PRF     = params.pulse_info.PRF;
    M       = params.pulse_info.M;
    t_start = params.pulse_info.t_start;

    T_chip = T_pulse / N_chips;          % chip duration (seconds)
    L_chip = round(T_chip * fs);         % samples per chip
    L_pulse = N_chips * L_chip;          % total samples per pulse (approx)

    % Build one pulse: phase per sample
    pulse_phase = zeros(L_pulse, 1);
    for chip_idx = 1:N_chips
        sample_range = (chip_idx-1)*L_chip + 1 : chip_idx*L_chip;
        if chip_idx <= length(code)
            pulse_phase(sample_range) = code(chip_idx);
        else
            pulse_phase(sample_range) = 0;   % fallback
        end
    end

    % Carrier phase contribution
    carrier_phase = 2*pi*fc*t;

    % Pulse train: sum over M pulses
    x = complex(zeros(N,1));
    for m = 1:M
        t_pulse_start = t_start + (m-1) / PRF;
        sample_start = round(t_pulse_start * fs) + 1;
        sample_end   = sample_start + L_pulse - 1;
        if sample_start < 1
            continue;
        end
        if sample_end > N
            sample_end = N;
            % truncate pulse if near end
            L_actual = sample_end - sample_start + 1;
            if L_actual <= 0
                continue;
            end
            pulse_phase_trunc = pulse_phase(1:L_actual);
        else
            pulse_phase_trunc = pulse_phase;
        end
        idx = sample_start:sample_end;
        x(idx) = x(idx) + exp(1i * (carrier_phase(idx) + pulse_phase_trunc + phi0));
    end

    % Apply amplitude
    x = A * x;

    % Normalise to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'PhaseCodedPulse: RMS is zero');
    x_clean = x / rms_val;

    % Assertions (standard)
    assert(iscolumn(x_clean), 'Output must be column.');
    assert(numel(x_clean) == spec.N, 'Length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex.');
    assert(~all(imag(x_clean(:)) == 0), 'Imag part must exist.');
    assert(all(isfinite(x_clean(:))), 'Inf/NaN found.');
end