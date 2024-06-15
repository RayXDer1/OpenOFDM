function [data_out] = ifft_trans(data_in,N_FFT)

    transpos_data = data_in.';
    ifft_data_out = ifft(transpos_data,N_FFT).*sqrt(N_FFT);
    data_out = ifft_data_out.';

end
