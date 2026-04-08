function x_clean = synthesize_clean_signal_class4(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS4
% Partial Band Noise Jamming with:
% - complex Gaussian noise
% - Butterworth bandpass filtering
% - upconversion to real RF
% - soft envelope clipping (PA saturation at RF)
% - downconversion to complex baseband
% - IQ imbalance (gain and phase mismatches)
% - carrier leakage (LO feedthrough)

    N  = double(spec.N);
    fs = double(spec.fs);
    t = (0:N-1)'/fs;

    fL = params.fL;
    fH = params.fH;
    alpha = params.alpha;
    theta = params.theta;
    delta = params.delta;
    filter_info = params.filter_info;

    % generate complex WGN
    n_w = (randn(N,1) + 1i*randn(N,1)) / sqrt(2);

    % design Butterworth BPF
    Wn = [fL, fH] / (fs/2);  % normalize to Nyquist

    assert(Wn(1) > 0 && Wn(2) < 1 && Wn(1) < Wn(2), ...
        'PBNJ: Invalid normalized band.');

    [b, a] = butter(filter_info.filter_order, Wn, 'bandpass');

    % apply zero-phase filtering
    x_f = filtfilt(b, a, n_w);

    % Up-conversion to RF (real signal)
    x_RF = real(x_f .* exp(1i * 2 * pi * params.fc * t));

    % PA soft clipping
    rms_RF = sqrt(mean(x_RF.^2));
    assert(rms_RF > 0, 'PBNJ: RMS is zero after filtering.');

    Vsat = filter_info.Vsat_factor * rms_RF;
    scale = min(1, Vsat ./ (abs(x_RF) + eps));
    x_RF_sat = x_RF .* scale;

    % Down-conversion to complex baseband
    x_BB = x_RF_sat .* exp(-1i * 2 * pi * params.fc * t);

    % add IQ imbalance
    I_imb = alpha * real(x_BB);
    Q_imb = sin(theta) * real(x_BB) + cos(theta) * imag(x_BB);
    x_IQ = I_imb + 1i * Q_imb;

    % carrier leakage (DC offset)
    x = x_IQ + delta;

    % amplitude
    x = params.A * x;

    % normalization to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'PBNJ: RMS is zero before normalization.');
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