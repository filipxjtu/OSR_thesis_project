function x_clean = synthesize_clean_signal_class6(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS6
% OFDM jamming with:
% - variable Nfft (64, 128, 256, 512)
% - cyclic prefix (1/16, 1/8, 1/4 of Nfft)
% - mixed BPSK/QPSK payload
% - variable active subcarrier ratio (60%-90% of Nfft)
% - symmetric active bins around DC
% - no conjugate symmetry
% - DC left empty
% - PA soft clipping
% - RF upconversion

    N  = double(spec.N);
    fs = double(spec.fs);

    A = params.A;
    fc = params.fc;
    phi = params.phi;
    ofdm_info = params.ofdm_info;

    Nfft = double(ofdm_info.Nfft);
    Lcp  = double(ofdm_info.Lcp);
    Nc   = double(ofdm_info.Nc);
    M    = double(ofdm_info.M);
    Nsym = double(ofdm_info.Nsym);
    Amax_factor = ofdm_info.Amax_factor;

    halfNc = Nc / 2;
    assert(mod(Nc,2) == 0, 'OFDM: Nc must be even.');
    assert(Nc<= Nfft-2, 'OFDM: Nc too large for Nfft');

    s_blocks = complex(zeros(M * Nsym, 1));
    idx = 1;

    for m = 1:M
        % generate payload symbols (randomly mix BPSK and QPSK)
        d = complex(zeros(Nc,1));

        for k = 1:Nc
            if rand > 0.5
                % BPSK
                b = 2 * randi([0,1]) - 1;   % {-1, +1}
                d(k) = complex(b, 0);
            else
                % QPSK
                re = 2 * randi([0,1]) - 1;
                im = 2 * randi([0,1]) - 1;
                d(k) = (re + 1i*im) / sqrt(2);
            end
        end

        % map into Nfft bins (leave DC empty, symmetric around DC)
        Xk = complex(zeros(Nfft,1));

        % Positive-frequency bins
        pos_bins = 2:(1 + halfNc);

        % Negative-frequency bins
        neg_bins = (Nfft - halfNc + 1):Nfft;

        Xk(pos_bins) = d(1:halfNc);
        Xk(neg_bins) = d(halfNc+1:end);

        %IFFT
        x_ifft = ifft(Xk, Nfft);

        % add cyclic prefix
        x_cp = [x_ifft(end-Lcp+1:end); x_ifft];

        %insert in the preallocated array
        s_blocks(idx: idx+Nsym-1) = x_cp;
        idx = idx + Nsym;
    end

    % truncate to exactly spec.N
    s_raw = s_blocks(1:N);

    % PA soft clipping
    rms_raw = sqrt(mean(abs(s_raw).^2));
    assert(rms_raw > 0, 'OFDM: RMS is zero before clipping.');

    Amax = Amax_factor * rms_raw;

    mag = abs(s_raw);
    scale = min(1, Amax ./ (mag + eps));
    s_clip = s_raw .* scale;

    % Upconvert
    t = (0:N-1)' / fs;
    x = A * s_clip .* exp(1i * (2 * pi * fc * t + phi));

    % Normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'OFDM: RMS is zero before normalization.');

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