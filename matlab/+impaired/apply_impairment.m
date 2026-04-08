function [x_imp, imp_params] = apply_impairment(x_clean, class_id, sample_index, spec, mode)
%APPLY_IMPAIRMENT Unified impairment layer for complex FANET jamming signals.
%
% Model:
%   y[n] = Norm( ADC_Quant( Rx_Imb( alpha * ( x[n] .* exp(j*theta[n]) ) * h[n] + w[n] ) ) )
%
% where:
%   - theta[n] includes residual CFO and Wiener phase noise
%   - h[n] is a 2-tap Ricean fading channel
%   - w[n] is complex AWGN
%   - Rx_Imb applies receiver DC offset and I/Q imbalance
%   - ADC_Quant applies 12-bit hardware clipping and quantization
%
%   - Output x_imp is RMS-normalized after all impairments
%   - SNR is defined before final normalization
%   - Same impairment family is used for all classes

    arguments
        x_clean {mustBeNumeric, mustBeVector}
        class_id {mustBeNumeric, mustBeInteger}
        sample_index (1,1) double {mustBeInteger, mustBeNonnegative}
        spec (1,1) struct
        mode {mustBeTextScalar}
    end

    % Shape handling
    x_shape_row = isrow(x_clean);
    x = x_clean(:);
    N = numel(x);

    assert(isfield(spec, 'dataset_seed'), 'apply_impairment:MissingSeed');
    assert(isfield(spec, 'N'), 'apply_impairment:MissingN');
    assert(N == double(spec.N), 'apply_impairment:LengthMismatch');
    assert(all(isfinite(real(x))) && all(isfinite(imag(x))), ...
        'apply_impairment:NonFiniteInput');

    mode = lower(string(mode));
    assert(mode == "train" || mode == "eval", 'apply_impairment:BadMode');

    % Defaults
    if ~isfield(spec, 'fs')
        error('apply_impairment:MissingFs', 'spec.fs is required.');
    end

    % SNR policy
    if ~isfield(spec, 'snr_mode')
        spec.snr_mode = "range";
    end
    if ~isfield(spec, 'snr_train_db')
        spec.snr_train_db = [-10, 5];
    end
    if ~isfield(spec, 'snr_eval_db')
        spec.snr_eval_db = [-5, 10];
    end
    if ~isfield(spec, 'snr_fixed_db')
        spec.snr_fixed_db = -6;
    end

    % Optional amplitude scaling
    if ~isfield(spec, 'enable_amp_scaling')
        spec.enable_amp_scaling = false;
    end
    if ~isfield(spec, 'amp_scale_range')
        spec.amp_scale_range = [0.95, 1.05];
    end

    % Residual CFO
    if ~isfield(spec, 'enable_cfo')
        spec.enable_cfo = true;
    end
    if ~isfield(spec, 'cfo_hz_range')
        spec.cfo_hz_range = [100, 500];
    end

    % Residual phase noise (smaller than emitter-level phase noise)
    if ~isfield(spec, 'enable_phase_noise')
        spec.enable_phase_noise = true;
    end
    if ~isfield(spec, 'phase_noise_std_range')
        spec.phase_noise_std_range = [0.001, 0.005];
    end

    % Ricean 2-tap channel
    if ~isfield(spec, 'enable_channel')
        spec.enable_channel = true;
    end
    if ~isfield(spec, 'channel_model')
        spec.channel_model = "ricean_2tap";
    end
    if ~isfield(spec, 'rice_k_db')
        spec.rice_k_db = 10;
    end
    if ~isfield(spec, 'delay_samp_range')
        spec.delay_samp_range = [1, 3];
    end
    if ~isfield(spec, 'echo_gain_db_range')
        spec.echo_gain_db_range = [-12, -6];
    end

    % Deterministic RNG
    base = uint32(mod(double(spec.dataset_seed), 2^32));
    class_idx_d = double(class_id) * 1000003;
    sample_idx_d = double(sample_index )* 1000033;

    unique_idx = (class_idx_d + sample_idx_d);
    mix_d = mod(1664525*unique_idx + 1013904223, 2^32);
    mix = uint32(mix_d);

    if mode == "train"
        mode_salt = uint32(314159265);
    else
        mode_salt = uint32(271828183);
    end

    imp_seed = double(bitxor(bitxor(base, mix), mode_salt));

    old_state = rng;
    rng(imp_seed, 'twister');


    % Input normalization safeguard
    rms_in = sqrt(mean(abs(x).^2));
    assert(isfinite(rms_in) && rms_in > 0, 'apply_impairment:BadInputRMS');
    x = x / rms_in;   

    % Target SNR selection
    snr_mode = lower(string(spec.snr_mode));

    switch snr_mode
        case "range"
            if mode == "train"
                snr_range = spec.snr_train_db;
            else
                snr_range = spec.snr_eval_db;
            end

            assert(numel(snr_range) == 2 && snr_range(1) < snr_range(2), ...
                'apply_impairment:BadSNRRange');

            target_snr_db = snr_range(1) + ...
                (snr_range(2) - snr_range(1)) * rand(1,1);

        case "fixed"
            target_snr_db = spec.snr_fixed_db;

        otherwise
            error('apply_impairment:BadSNRMode', ...
                'snr_mode must be "range" or "fixed".');
    end

    % Optional amplitude scaling
    amp_scale = 1.0;
    if spec.enable_amp_scaling
        r = spec.amp_scale_range;
        assert(numel(r) == 2 && r(1) > 0 && r(2) > r(1), ...
            'apply_impairment:BadAmpRange');

        amp_scale = r(1) + (r(2) - r(1)) * rand(1,1);
    end

    x_tx = amp_scale * x;


    % Residual CFO + phase noise
    fs = double(spec.fs);
    n = (0:N-1)';

    cfo_hz = 0.0;
    if spec.enable_cfo
        cfo_mag = spec.cfo_hz_range(1) + ...
                  diff(spec.cfo_hz_range) * rand(1,1);
        cfo_sign = 2 * (rand > 0.5) - 1;
        cfo_hz = cfo_sign * cfo_mag;
    end

    sigma_pn_imp = 0.0;
    theta_pn = zeros(N,1);

    if spec.enable_phase_noise
        sigma_pn_imp = spec.phase_noise_std_range(1) + ...
            diff(spec.phase_noise_std_range) * rand(1,1);

        % wiener increments
        w_pn = sigma_pn_imp * randn(N,1) * sqrt(1/fs);

        % continious wrapping
        theta_pn = mod(cumsum(w_pn) + pi, 2 * pi) - pi;
    end

    theta_cfo = 2*pi*cfo_hz * (n / fs);
    theta_total = theta_cfo + theta_pn;

    x_osc = x_tx .* exp(1i * theta_total);

    % Shared channel model
    delay_samp = 0;
    echo_gain_db = 0;
    rice_k_db = 0;
    h = 1;

    if spec.enable_channel
        assert(lower(string(spec.channel_model)) == "ricean_2tap", ...
            'apply_impairment:OnlyRicean2TapSupported');

        delay_samp = randi(spec.delay_samp_range);
        echo_gain_db = spec.echo_gain_db_range(1) + ...
            diff(spec.echo_gain_db_range) * rand(1,1);
        rice_k_db = spec.rice_k_db;

        Klin = 10^(rice_k_db/10);
        a_los = sqrt(Klin / (Klin + 1));
        a_echo = sqrt(1 / (Klin + 1)) * 10^(echo_gain_db/20);

        echo_phase = 2*pi*rand(1,1);

        h = zeros(delay_samp + 1, 1);
        h(1) = a_los;
        h(end) = a_echo * exp(1i * echo_phase);

        % normalize channel energy
        h = h / sqrt(sum(abs(h).^2));

        %provide intial conditions
        zi = filtic(h,1, x_osc(1)*ones(delay_samp,1));
        
        x_ch = filter(h, 1, x_osc, zi);
    else
        x_ch = x_osc;
    end

    % AWGN injection
    P_signal = mean(abs(x_ch).^2);
    assert(isfinite(P_signal) && P_signal > 0, ...
        'apply_impairment:BadSignalPower');

    noise_variance = P_signal * 10^(-target_snr_db / 10);
    assert(isfinite(noise_variance) && noise_variance >= 0, ...
        'apply_impairment:BadNoiseVar');

    n_awgn = sqrt(noise_variance/2) * (randn(N,1) + 1i*randn(N,1));
    x_noisy = x_ch + n_awgn;

    P_noise_realized = mean(abs(x_noisy - x_ch).^2);
    realized_snr_db = 10*log10(P_signal / max(P_noise_realized, eps));

    % receiver I/Q Imbalance (Generic slight mismatch)
    rx_alpha = 1.0 + (rand()-0.5)*0.05;  % +/- 2.5% amplitude imbalance
    rx_theta = (rand()-0.5)*0.087;       % +/- 5 degrees phase imbalance
    I_rx = rx_alpha * real(x_noisy);
    Q_rx = sin(rx_theta)*real(x_noisy) + cos(rx_theta)*imag(x_noisy);
    x_rx = I_rx + 1i*Q_rx;

    % receiver DC Offset (LO Feedthrough)
    rx_dc = (randn() + 1i*randn()) * 0.01; % small constant leakage
    x_rx = x_rx + rx_dc;
    
    % fixed reciever scalling to make headroom before ADC
    rx_scale = 0.30;   % typical range(0.25 to 0.35)
    x_rx_scaled = x_rx * rx_scale;    

    % ADC Quantization (12-bit SDR simulation)
    bit_depth = 12;
    L = 2^(bit_depth - 1) - 1; 

    % Clip to [-1, 1] range to simulate ADC saturation
    x_rx_clipped = complex(max(-1, min(1, real(x_rx_scaled))), max(-1, min(1, imag(x_rx_scaled))));

    % Quantize
    x_rx_quant = round(x_rx_clipped * L) / L;

    % calculate rms
    rms_out = sqrt(mean(abs(x_rx_quant).^2));
    assert(isfinite(rms_out) && rms_out > 0, ...
        'apply_impairment:BadOutputRMS');

    x_imp_col = x_rx_quant;

    % restore original orientation
    if x_shape_row
        x_imp = x_imp_col.';
    else
        x_imp = x_imp_col;
    end

    assert(all(isfinite(real(x_imp))) && all(isfinite(imag(x_imp))), ...
        'apply_impairment:NonFiniteOutput');

    % Pack impairment parameters
    imp_params = impaired.init_imp_param_record();

    imp_params.impairment_seed   = imp_seed;
    imp_params.snr_mode          = char(snr_mode);
    imp_params.target_snr_db     = target_snr_db;
    imp_params.realized_snr_db   = realized_snr_db;
    imp_params.noise_variance    = noise_variance;
    imp_params.P_signal          = P_signal;
    imp_params.P_noise_realized  = P_noise_realized;
    imp_params.amp_scale         = amp_scale;

    imp_params.rms_out = rms_out;

    imp_params.cfo_hz            = cfo_hz;
    imp_params.phase_noise_std   = sigma_pn_imp;
    imp_params.delay_samp        = delay_samp;
    imp_params.echo_gain_db      = echo_gain_db;
    imp_params.rice_k_db         = rice_k_db;
    imp_params.channel_energy    = sum(abs(h).^2);
    
    %debug_dir = fullfile('artifacts', 'debug');
    %if ~exist(debug_dir, 'dir'), mkdir(debug_dir); end
    %debug_file = fullfile(debug_dir, sprintf('rng_debug_seed%d.json', double(spec.dataset_seed)));
    
    %debug_info = struct(...
    %    'class_id', double(class_id), ...
    %    'sample_index', double(sample_index), ...
    %    'generated_seed', imp_seed, ...
    %    'target_snr', target_snr_db ...
    %);
    
    %fid = fopen(debug_file, 'a');
    %fprintf(fid, '%s\n', jsonencode(debug_info));
    %fclose(fid);
  
    % Restore RNG
    rng(old_state);
end