function [cfo_estimate_value,data_out] = cfo_estimate(data_in,data_preamble,Fs,N_FFT)

% 3种方法的性能对比 1 < 2 = 3
%% 1.前后一段LTF CFO估计精度较差 估计范围: 正负156.25KHz
% % delay
% D = N_FFT;
% 
% % 时域估计方法
% DATA1 = data_preamble(1:D);
% DATA2 = data_preamble(1*D+1:2*D);
% K = 1;

%% 2.前后间隔2个LTF估计 有助于提升CFO估计精度 估计范围: 正负78.125KHz
% delay
D = N_FFT;

% 时域估计方法
DATA1 = data_preamble(1:D);
DATA2 = data_preamble(3*D+1:4*D);
K = 3;

%% 3.将两段LTF视为一段LTF 有助于提升CFO估计精度 估计范围: 正负52.0833KHz
% % delay
% D = N_FFT*2;
% 
% % 时域估计方法
% DATA1 = data_preamble(1:D);
% DATA2 = data_preamble(1*D+1:2*D);
% K = 2;

%% 估计频偏值并进行补偿和校正
% 使用延迟自相关算法计算频偏值
x_corr = sum(DATA1.*conj(DATA2));

% 计算出频偏值
cfo_estimate_value = -angle(x_corr)/K*(Fs)/(2*pi*N_FFT);
% cfo_estimate_value = 100;

% 根据计算出的频偏值进行补偿和校正
timebase = (0:length(data_in)-1);
data_out = exp(1j*(2*pi*(-cfo_estimate_value)*timebase/Fs)).*data_in;

end
