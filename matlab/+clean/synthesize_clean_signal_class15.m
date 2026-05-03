function x_clean = synthesize_clean_signal_class15(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS16
% Reactive Burst Jammer (protocol‑aware, narrowband pulses)
%
% Model:
%   x(t) = sum_{k=1}^{M} A_k * w_k(t - t_k) * exp(j*2*pi*fc*t)
%   where w_k(t) is a short burst of narrowband white Gaussian noise,
%   duration T_on (samples), bandwidth B (Hz), centre frequency fc_maybe.
%   Inter-arrival times are drawn from a log-normal distribution.
%
% Discriminative features:
%   - extremely high envelope kurtosis (>10)
%   - narrow bandwidth (unlike PGPJ which is full‑band bursts)
%   - bursty but not periodic (unlike PGPJ's regular train)
%
% Parameters:
%   A           : global amplitude (linear)
%   fc          : carrier centre frequency (Hz)
%   B           : bandwidth of each burst (Hz) – narrow (e.g., 100-500 kHz)
%   T_on_mean   : mean burst duration (samples)
%   T_on_std    : std of burst duration (samples)
%   T_off_lognorm_mean : mean of log(inter-arrival) (scale parameter)
%   T_off_lognorm_std  : std of log(inter-arrival) (shape parameter)
%   M_max       : maximum number of bursts to generate
%
% Output:
%   x_clean     : column vector, complex, unit RMS

    N  = double(spec.N);
    fs = double(spec.fs);

    A        = params.A;
    fc       = params.fc;
    B        = params.burst_info.B;
    T_on_mean= params.burst_info.T_on_mean;
    T_on_std = params.burst_info.T_on_std;
    mu_ln    = params.burst_info.mu_ln;      % mean of log(interval)
    sigma_ln = params.burst_info.sigma_ln;
    M_max    = params.burst_info.M_max;

    x = zeros(N, 1);

    % Generate burst start times using log-normal inter-arrival
    t_start = 0;
    burst_count = 0;
    while t_start < (N/fs) && burst_count < M_max
        % Duration of this burst (samples) – truncated normal
        T_on_samples = max(1, round(T_on_mean + T_on_std * randn));
        T_on_samples = min(T_on_samples, N - round(t_start * fs) - 1);
        if T_on_samples <= 0
            break;
        end

        % Band-limited noise for this burst (narrowband)
        % Design a simple bandpass filter: Gaussian white noise -> BPF
        n_extra = 100;
        n_wgn = (randn(T_on_samples + n_extra, 1) + 1i*randn(T_on_samples + n_extra, 1)) / sqrt(2);
        % Butterworth bandpass of order 4
        Wn = [fc - B/2, fc + B/2] / (fs/2);
        Wn = max(0, min(1, Wn));
        if Wn(1) >= Wn(2) || Wn(2) <= 0 || Wn(1) >= 1
            % fallback to white noise if band invalid
            burst = n_wgn(1:T_on_samples);
        else
            [b, a] = butter(4, Wn, 'bandpass');
            n_filt = filter(b, a, n_wgn);
            burst = n_filt(n_extra+1 : n_extra+T_on_samples);
        end

        % Amplitude scaling (optional per‑burst variation)
        A_burst = A * (0.8 + 0.4 * rand);   % 0.8–1.2

        % Place into signal vector
        start_idx = round(t_start * fs) + 1;
        end_idx = min(start_idx + T_on_samples - 1, N);
        burst = burst(1:end_idx - start_idx + 1);
        x(start_idx:end_idx) = x(start_idx:end_idx) + A_burst * burst;

        % Next start time: sample inter-arrival from log-normal
        inter_arrival = exp(mu_ln + sigma_ln * randn);  % seconds
        t_start = t_start + inter_arrival;
        burst_count = burst_count + 1;
    end


    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'ReactiveBurst: RMS zero');
    x_clean = x / rms_val;

    % Assertions
    assert(iscolumn(x_clean), 'Output must be column.');
    assert(numel(x_clean) == spec.N, 'Length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex.');
    assert(~all(imag(x_clean(:)) == 0), 'Imag part must exist.');
    assert(all(isfinite(x_clean(:))), 'Inf/NaN found.');
end