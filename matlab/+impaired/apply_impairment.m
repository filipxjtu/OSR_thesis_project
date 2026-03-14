function [x_imp, imp_params] = apply_impairment(x_clean, sample_index, spec, mode)
%APPLY_IMPAIRMENT Deterministic impairment layer (v1).
%
% Inputs
%   x_clean       : (N x 1) or (1 x N) real vector
%   sample_index  : integer >= 0 (deterministic selector)
%   spec          : struct with fields:
%                  - dataset_seed (scalar integer)
%                  - fs, N (required elsewhere; N enforced here)
%                  - snr_train_db = [min max] (optional)
%                  - snr_eval_db  = [min max] (optional)
%                  - enable_phase_offset (optional, default false)
%                  - enable_amp_scaling  (optional, default false)
%                  - phase_range_rad     (optional, default [-pi pi])
%                  - amp_scale_range     (optional, default [0.8 1.2])
%   mode          : "train" or "eval"
%
% Outputs
%   x_imp         : impaired + normalized signal, same shape as input
%   imp_params    : struct (traceability + realized metrics)

    arguments
        x_clean {mustBeNumeric, mustBeVector}
        sample_index (1,1) double {mustBeInteger, mustBeNonnegative}
        spec (1,1) struct
        mode (1,:) char
    end

    % Standardize shape (internal)
    x_shape_row = isrow(x_clean);
    x = x_clean(:); % column
    N = numel(x);

    x = x - mean(x);

    if isfield(spec, 'N')
        assert(N == spec.N, 'apply_impairment:LengthMismatch', ...
            'x_clean length (%d) must equal spec.N (%d).', N, spec.N);
    end

    assert(all(isfinite(x)), 'apply_impairment:NonFiniteInput', 'x_clean contains NaN/Inf.');
    assert(isreal(x), 'apply_impairment:ComplexInput', 'x_clean must be real (v1 contract).');

    % Policy defaults
    if ~isfield(spec, 'snr_train_db'), spec.snr_train_db = [-12 -2]; end
    if ~isfield(spec, 'snr_eval_db'),  spec.snr_eval_db  = [-2 8]; end

    if ~isfield(spec, 'enable_phase_offset'), spec.enable_phase_offset = false; end
    if ~isfield(spec, 'enable_amp_scaling'),  spec.enable_amp_scaling  = false; end

    if ~isfield(spec, 'phase_range_rad'), spec.phase_range_rad = [-pi pi]; end
    if ~isfield(spec, 'amp_scale_range'), spec.amp_scale_range = [0.8 1.2]; end

    mode = lower(string(mode));
    if mode == "train"
        snr_range = spec.snr_train_db;
    elseif mode == "eval"
        snr_range = spec.snr_eval_db;
    else
        error('apply_impairment:BadMode', 'mode must be "train" or "eval".');
    end
    assert(numel(snr_range)==2 && snr_range(1) < snr_range(2), ...
        'apply_impairment:BadSNRRange', 'SNR range must be [min max] with min<max.');

    % Deterministic RNG seed for impairment
    assert(isfield(spec, 'dataset_seed'), 'apply_impairment:MissingSeed', 'spec.dataset_seed required.');
    
    % Use a simple, stable mixing formula
    base = uint32(mod(double(spec.dataset_seed), 2^32));
    idx  = uint32(mod(sample_index, 2^32));
    mix = uint32(1664525) .* idx + uint32(1013904223);
    imp_seed = double(bitxor(base, mix)); % LCG-like mix

    old_state = rng;
    rng(imp_seed, 'twister');

    % Draw target SNR uniformly from policy range
    target_snr_db = snr_range(1) + (snr_range(2) - snr_range(1)) * rand(1,1);

    % Optional amplitude scaling (applied before AWGN)
    amp_scale = 1.0;
    if spec.enable_amp_scaling
        r = spec.amp_scale_range;
        assert(numel(r)==2 && r(1) > 0 && r(2) > r(1), 'apply_impairment:BadAmpRange', ...
            'amp_scale_range must be [min max] with 0<min<max.');
        amp_scale = r(1) + (r(2)-r(1)) * rand(1,1);
        x = amp_scale * x;
    end

    % ---- Optional global phase offset ----
    % NOTE: With a real-only v1 contract, a "phase offset" is ambiguous.
    % Implemented as a deterministic sign/phase-like operation by mixing with Hilbert analytic phase,
    % but that yields complex. So for v1 we implement a *global polarity flip* as the only real-safe "phase".
    % If you later allow complex IQ, replace this block with x = real(x .* exp(1j*phi)).
    phase_offset_rad = 0.0;
    if spec.enable_phase_offset
        r = spec.phase_range_rad;
        assert(numel(r)==2 && r(2) > r(1), 'apply_impairment:BadPhaseRange', ...
            'phase_range_rad must be [min max] with max>min.');
        phase_offset_rad = r(1) + (r(2)-r(1)) * rand(1,1);

        %Real-safe proxy
        if cos(phase_offset_rad) < 0
            x = -x;
        end
    end

    % Compute signal power
    P_signal = mean(x.^2);
    assert(isfinite(P_signal) && P_signal > 0, 'apply_impairment:BadSignalPower', ...
        'Signal power must be finite and > 0.');

    % Noise variance for target SNR
    noise_variance = P_signal * 10^(-target_snr_db/10);
    assert(isfinite(noise_variance) && noise_variance >= 0, 'apply_impairment:BadNoiseVar', ...
        'Noise variance must be finite and >=0.');

    % Add AWGN
    n = sqrt(noise_variance) * randn(N,1);
    x_awgn = x + n;

    % Realized SNR (measured vs clean-after-optional-scaling/polarity)
    P_noise_realized = mean((x_awgn - x).^2);
    realized_snr_db = 10*log10(P_signal / max(P_noise_realized, eps));

   % v1 normalization: normalize using CLEAN (pre-noise) RMS, not impaired RMS
    rms_clean = sqrt(mean(x.^2));
    assert(isfinite(rms_clean) && rms_clean > 0, 'apply_impairment:BadRMS', 'Clean RMS must be finite and >0.');

    x_norm = x_awgn / rms_clean;

    % store for traceability
    rms_val = rms_clean;

    tol_db = 1.0;
    assert(abs(realized_snr_db - target_snr_db) < tol_db, ...
        'apply_impairment:SNRDeviation', ...
        'Realized SNR deviates too much from target.');

    % Output shape restore
    if x_shape_row
        x_imp = x_norm.'; % row
    else
        x_imp = x_norm;   % column
    end

    assert(all(isfinite(x_imp)), 'apply_impairment:NonFiniteOutput', 'x_imp contains NaN/Inf.');

    % Pack parameters
    imp_params = impaired.init_imp_param_record();
    imp_params.impairment_seed   = imp_seed;
    imp_params.target_snr_db     = target_snr_db;
    imp_params.realized_snr_db   = realized_snr_db;
    imp_params.noise_variance    = noise_variance;
    imp_params.P_signal          = P_signal;
    imp_params.P_noise_realized  = P_noise_realized;
    imp_params.amp_scale         = amp_scale;
    imp_params.phase_offset_rad  = phase_offset_rad;
    imp_params.normalization_rms = rms_val;


    rng(old_state);
end
