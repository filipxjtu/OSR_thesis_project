function x_clean = synthesize_clean_signal_class13(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS13
% Phase-Coded Pulse Jammer (Barker-13 or true P4 polyphase)
%
% A pulse train where each pulse's carrier phase is modulated by a
% discrete code sequence over N_chips chips of equal duration. Integer
% chip length is computed in the param sampler so L_pulse = N_chips *
% L_chip exactly, avoiding drift between requested and realized T_pulse.
%
% Discriminative features (post-RX-chain):
%   - rectangular pulse envelope (vs. Gaussian PGPJ)
%   - discrete phase-step PMF (vs. continuous-phase LFM/SFM)
%   - wide instantaneous bandwidth (vs. CW)

    N  = double(spec.N);
    fs = double(spec.fs);
    t  = (0:N-1)' / fs;

    A        = params.A;
    fc       = params.fc;
    phi0     = params.phi;
    code     = params.pulse_info.code_phase;
    N_chips  = params.pulse_info.N_chips;
    L_chip   = params.pulse_info.L_chip;
    L_pulse  = params.pulse_info.L_pulse;
    PRF      = params.pulse_info.PRF;
    M        = params.pulse_info.M;
    t_start  = params.pulse_info.t_start;

    % Build per-sample phase vector for ONE pulse (length L_pulse)
    pulse_phase = zeros(L_pulse, 1);
    for chip_idx = 1:N_chips
        rng_lo = (chip_idx - 1) * L_chip + 1;
        rng_hi = chip_idx * L_chip;
        pulse_phase(rng_lo:rng_hi) = code(chip_idx);
    end

    % Carrier phase across the entire window
    carrier_phase = 2*pi*fc*t;

    % Sum M pulses into the output
    x = complex(zeros(N, 1));
    for m = 1:M
        t_pulse_start = t_start + (m - 1) / PRF;
        sample_start = round(t_pulse_start * fs) + 1;
        if sample_start > N
            break;
        end
        sample_end = sample_start + L_pulse - 1;

        if sample_end > N
            % Pulse runs past window: truncate cleanly
            L_actual = N - sample_start + 1;
            if L_actual < 1
                continue;
            end
            idx = sample_start:N;
            phase_chunk = pulse_phase(1:L_actual);
        else
            idx = sample_start:sample_end;
            phase_chunk = pulse_phase;
        end

        x(idx) = x(idx) + exp(1i * (carrier_phase(idx) + phase_chunk + phi0));
    end

    x = A * x;

    % RMS normalize
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'PhaseCodedPulse: RMS is zero before normalization.');
    x_clean = x / rms_val;

    % Assertions
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex.');
    assert(~all(imag(x_clean(:)) == 0), ...
        'Signal must have non-zero imaginary component.');
    assert(all(isfinite(x_clean(:))), 'Signal contains NaN/Inf.');
end