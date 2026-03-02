function fields = get_active_fields(class_id)
    %GET_ACTIVE_FIELDS Returns required parameter fields per class.
    %
    % This defines which fields in params must be populated (non-NaN / non-empty)
    % for each clean signal class.

    class_id = int32(class_id);

    switch class_id

        case 0  % Single Tone
            fields = ["A","f0","phi"];

        case 1  % Multi-Tone
            fields = ["A","f0","phi","K"];

        case 2  % LFM
            fields = ["A","f0","f1","phi","T"];

        case 3  % SFM
            fields = ["A","fc","df","fm","beta","phi"];

        case 4  % PBN
            fields = ["A","fL","fH", "bandwidth", ...
                "bin_info.nfft","bin_info.freq_res", ...
                "bin_info.bin_L","bin_info.bin_H","bin_info.phi_bins"];

        case 5  % Noise FM
            fields = ["A","fc","kappa","phi","fL","fH", "bandwidth", ...
                "bin_info.nfft","bin_info.freq_res", ...
                "bin_info.bin_L","bin_info.bin_H","bin_info.phi_bins"];

        case 6  % Frequency Hopping
            fields = ["A", "hop_info.Lhop", "hop_info.hop_set", ...
                "hop_info.n_hops", "hop_info.hop_idx", "hop_info.phi_hops"];

        otherwise
            error("Invalid class_id: %d", class_id);

    end
end