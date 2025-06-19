function [data_out] = channel_compen(data_in,chan_estim_vaule)
%倍采样条件
[N_FFT,num] = size(data_in);
data_out=zeros(N_FFT,num);
a = real(chan_estim_vaule.');
b = imag(chan_estim_vaule.');

for k = 1 : num

    DATA = data_in(:,k);
    c = real(DATA);
    d = imag(DATA);
    DATA_real = a .* c - b .* d;
    DATA_imag = a .* d + b .* c;
    data_out(:,k) = DATA_real + 1i .* DATA_imag;

end

% MSE_chan_est = (data_out()-data)^2/(data^2);
end