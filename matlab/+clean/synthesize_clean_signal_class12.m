function x_clean = synthesize_clean_signal_class12(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS12
% Reactive Burst Jammer (protocol-aware, narrowband bursts)
%
% Bursty narrowband-noise jammer modeling a listen-then-strike attacker.
% Each burst is band-limited Gaussian noise centered at fc with bandwidth
% B; bursts are gated by a raised-cosine envelope to suppress out-of-band
% splatter. Inter-arrival times follow a log-normal distribution.
%
% Discriminative features (post-RX-chain):
%   - high envelope kurtosis (bursty)
%   - narrow spectral support (unlike PGNJ which is full-band bursts)
%   - aperiodic arrivals (unlike PGPJ's regular train)

    N  = double(spec.N);
    fs = double(spec.fs);

    A         = params.A;
    fc        = params.fc;
    B         = params.burst_info.B;
    T_on_mean = params.burst_info.T_on_mean;
    T_on_std  = params.burst_info.T_on_std;
    mu_ln     = params.burst_info.mu_ln;
    sigma_ln  = params.burst_info.sigma_ln;
    M_max     = params.burst_info.M_max;

    x = complex(zeros(N, 1));

    % Bandpass filter shared across bursts (deterministic given fs, fc, B)
    Wn = [fc - B/2, fc + B/2] / (fs/2);
    Wn = max(1e-6, min(1 - 1e-6, Wn));
    use_bpf = (Wn(1) < Wn(2)) && (Wn(1) > 0) && (Wn(2) < 1);
    if use_bpf
        [b_bpf, a_bpf] = butter(4, Wn, 'bandpass');
    end

    t_start_sec  = 0;
    burst_count  = 0;
    n_extra      = 100;   % filter warm-up samples to discard

    while t_start_sec < (N/fs) && burst_count < M_max
        % Burst duration
        T_on_samples = max(1, round(T_on_mean + T_on_std * randn));
        start_idx = round(t_start_sec * fs) + 1;
        if start_idx > N
            break;
        end
        T_on_samples = min(T_on_samples, N - start_idx + 1);
        if T_on_samples < 4   % too short to taper meaningfully
            t_start_sec = t_start_sec + exp(mu_ln + sigma_ln * randn);
            burst_count = burst_count + 1;
            continue;
        end

        % Generate band-limited complex noise for this burst
        n_wgn = (randn(T_on_samples + n_extra, 1) + ...
                 1i * randn(T_on_samples + n_extra, 1)) / sqrt(2);
        if use_bpf
            n_filt = filter(b_bpf, a_bpf, n_wgn);
            burst = n_filt(n_extra + 1 : n_extra + T_on_samples);
        else
            burst = n_wgn(1:T_on_samples);
        end

        % Raised-cosine taper at burst edges to suppress spectral splatter
        ramp_len = min(8, floor(T_on_samples / 4));
        if ramp_len > 0
            ramp = (1 - cos(pi * (0:ramp_len-1)' / ramp_len)) / 2;
            burst(1:ramp_len) = burst(1:ramp_len) .* ramp;
            burst(end-ramp_len+1:end) = burst(end-ramp_len+1:end) .* flipud(ramp);
        end

        % Per-burst amplitude variation
        A_burst = A * (0.8 + 0.4 * rand);

        % Place into output
        end_idx = start_idx + T_on_samples - 1;
        x(start_idx:end_idx) = x(start_idx:end_idx) + A_burst * burst;

        % Next start time: log-normal inter-arrival
        inter_arrival_sec = exp(mu_ln + sigma_ln * randn);
        t_start_sec = t_start_sec + inter_arrival_sec;
        burst_count = burst_count + 1;
    end

    % RMS normalize
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'ReactiveBurst: RMS is zero before normalization.');
    x_clean = x / rms_val;

    % Assertions
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex.');
    assert(~all(imag(x_clean(:)) == 0), ...
        'Signal must have non-zero imaginary component.');
    assert(all(isfinite(x_clean(:))), 'Signal contains NaN/Inf.');
end