function [chan_estim_vaule4] = channel_estim(data_in,loc_L)
% ----------------------输入参数处理----------------------
N = length(loc_L);
L_fft=loc_L;
a = real(L_fft);
b = imag(L_fft);


% ----------------------接收信号----------------------------
Y1 = data_in(1:N);
Y2 = data_in(N+1:N*2);
Y_average = 1/2 .* (Y1 + Y2);
c = real(Y_average);
d = imag(Y_average);

% ----------------------信道估计----------------------------
A1 = c .^2;
A2 = d .^2;
A = A1 + A2;
B = a .*  c + b .* d ;
C = b .*  c - a .* d;
H_real = B ./ A;
H_imag = C ./ A;
   

chan_estim_vaule4 = H_real + 1i * H_imag;

% figure;plot(real(data_compen));
% hold on;
% plot(real(data));

end