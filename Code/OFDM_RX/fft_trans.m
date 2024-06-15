function [data_out] = fft_trans(data_in,N_FFT)

    transpos_data = data_in.';
    fft_data_out = fft(transpos_data,N_FFT)./sqrt(N_FFT);
    data_out = fft_data_out.';

end
