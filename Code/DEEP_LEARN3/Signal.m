clc;
clear all;

speed = [50 150 250 350 450];                 % 移动速度，单位km/h
M = 4;

N = 20;                       % 子载波数量
v = speed * 1000 / 3600;      % 移动速度单位转换成m/s
c = 3e8;                      % 光速
S = 5;                        % OFDM符号数

snr = [5 10 15 20 25];

fc = 2e9;                     % 载波频率，默认值900MHz
fd = v * fc / c;
dt = 1;

%生成待测试信号数据

num = 10000;                      % 每个SNR待测试数据数量
p = 1 + 0*1i;                  % 导频暂时设置为恒定值1 + 0*j；

H_s = zeros(5 * length(snr) * num,N);
H_p = zeros(5 * length(snr) * num,N);
X_p = zeros(5 * length(snr) * num,N);
Y_p = zeros(5 * length(snr) * num,N);

H = zeros(4 * num * size(fd,2),N);            % 生成的信道数据

Max_0 = 0;
for i = 1 : 1 : 4 * num
    for j = 1 : 1 : 5
    h = Jakes_Rayleigh(fd(j), M, dt, N);           % 修改后给出的是一个复数
    H_1 = fft(h);
    Max = max(max(max(abs(real(H_1)))),max(max(abs(imag(H_1)))));
    Max_0 = max(Max_0,Max);
    I = (i - 1) * 5 + j;
    H(I,:) = H_1;
    end
end
H = H ./ Max_0;

%信道矩阵分割

MAX_3 = 0;    
for j = 1 : 1 : length(snr)
    
    X_data = zeros(20 * num,N);
    %将导频插入到实际数据中,只有前四个符号存在导频

    for n = 1 : 1 : 5 * num
        %前半部分符号
        X_b = zeros(4,N);
        X_b_r = sign(randn(4,15));                
        X_b_i = sign(randn(4,15));       
        X_1 = X_b_r + 0i * X_b_i;

        for i = 1 : 1 : 4
            for k = 1 : 1 : 5
                if(i == 1)
                    X_b(i,4 * (k - 1) + 1) = p;
                    X_b(i,4 * (k - 1) + 2) = X_1(i,3 * (k - 1) + 1);
                    X_b(i,4 * (k - 1) + 3) = X_1(i,3 * (k - 1) + 2);
                    X_b(i,4 * (k - 1) + 4) = X_1(i,3 * (k - 1) + 3);
                elseif(i == 2)
                    X_b(i,4 * (k - 1) + 1) = X_1(i,3 * (k - 1) + 1);
                    X_b(i,4 * (k - 1) + 2) = p;
                    X_b(i,4 * (k - 1) + 3) = X_1(i,3 * (k - 1) + 2);
                    X_b(i,4 * (k - 1) + 4) = X_1(i,3 * (k - 1) + 3);
                elseif(i == 3)
                    X_b(i,4 * (k - 1) + 1) = X_1(i,3 * (k - 1) + 1);
                    X_b(i,4 * (k - 1) + 2) = X_1(i,3 * (k - 1) + 2);
                    X_b(i,4 * (k - 1) + 3) = p;
                    X_b(i,4 * (k - 1) + 4) = X_1(i,3 * (k - 1) + 3);
                elseif(i == 4)
                    X_b(i,4 * (k - 1) + 1) = X_1(i,3 * (k - 1) + 1);
                    X_b(i,4 * (k - 1) + 2) = X_1(i,3 * (k - 1) + 2);
                    X_b(i,4 * (k - 1) + 3) = X_1(i,3 * (k - 1) + 3);
                    X_b(i,4 * (k - 1) + 4) = p;
                end
            end
        end
        X_data(4 * (n - 1) + 1,:) = X_b(1,:);
        X_data(4 * (n - 1) + 2,:) = X_b(2,:);
        X_data(4 * (n - 1) + 3,:) = X_b(3,:);
        X_data(4 * (n - 1) + 4,:) = X_b(4,:);
    end
    
    %添加信道
    Y_2 = zeros(20 * num,N);
    Y_data = zeros(20 * num,N);
    
    Y_1 = X_data .* H;
    for i = 1 : 1 : 20 * num
        Y_2(i,:) = ifft(Y_1(i,:));
    end
    
    %添加噪声
    Y = awgn(Y_2,snr(1,j));
    
    for i = 1 : 1 : 20 * num
        Y_data(i,:) = fft(Y(i,:));
    end
    
    H_LS_1 = Y_data ./ X_data;

    MAX_1 = max(max(abs(real(H_LS_1))));
    MAX_2 = max(max(abs(imag(H_LS_1))));
    MAX_z = max(MAX_1,MAX_2);
    MAX_3 = max(MAX_3,MAX_z);

    %抽取导频信号处发送、接收信号及估计值
    H_s_1 = zeros(5 * num,N);
    H_p_1 = zeros(5 * num,N);
    X_p_1 = zeros(5 * num,N);
    Y_p_1 = zeros(5 * num,N);
    
    for i = 1 : 1 : 5 * num
        for k = 1 : 4 : 17
            I = 4 * (i - 1);
            H_s_1(i,k) = H(I + 1,k);
            H_s_1(i,k + 1) = H(I + 2,k + 1);
            H_s_1(i,k + 2) = H(I + 3,k + 2);
            H_s_1(i,k + 3) = H(I + 4,k + 3);

            H_p_1(i,k) = H_LS_1(I + 1,k);
            H_p_1(i,k + 1) = H_LS_1(I + 2,k + 1);
            H_p_1(i,k + 2) = H_LS_1(I + 3,k + 2);
            H_p_1(i,k + 3) = H_LS_1(I + 4,k + 3);
    
            X_p_1(i,k) = X_data(I + 1,k);
            X_p_1(i,k + 1) = X_data(I + 2,k + 1);
            X_p_1(i,k + 2) = X_data(I + 3,k + 2);
            X_p_1(i,k + 3) = X_data(I + 4,k + 3);
    
            Y_p_1(i,k) = Y_data(I + 1,k);
            Y_p_1(i,k + 1) = Y_data(I + 2,k + 1);
            Y_p_1(i,k + 2) = Y_data(I + 3,k + 2);
            Y_p_1(i,k + 3) = Y_data(I + 4,k + 3);
        end
    end
    

    for i= 1 : 1 : 5 * num
        H_p(5 * num * (j - 1) + i,:) = H_p_1(i,:);
        H_s(5 * num * (j - 1) + i,:) = H_s_1(i,:);
        X_p(5 * num * (j - 1) + i,:) = X_p_1(i,:);
        Y_p(5 * num * (j - 1) + i,:) = Y_p_1(i,:);
    end

end

H_p = H_p / MAX_3;

% A_1 = real(H_p); A_2 = imag(H_p);
% B_1 = real(H_s); B_2 = imag(H_s);
% 
% A_1 = A_1'; A_2 = A_2';
% B_1 = B_1'; B_2 = B_2';
% 
% error1 = sqrt(sum((A_1 - B_1).^2) ./ N);
% error2 = sqrt(sum((A_2 - B_2).^2) ./ N);
% figure(1)
% subplot(2,1,1)
% plot(error1)
% xlabel('序列点数')
% ylabel('LS实部均方根误差')
% string = {'LS实部结果对比'};
% title(string)
% grid
% 
% subplot(2,1,2)
% plot(error2)
% xlabel('序列点数')
% ylabel('LS虚部均方根误差')
% string = {'LS虚部结果对比'};
% title(string)
% grid

%导频估计数据集
writematrix(real(H_s(1:50000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s1_r.xls');
writematrix(real(H_s(50001:100000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s2_r.xls');
writematrix(real(H_s(100001:150000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s3_r.xls');
writematrix(real(H_s(150001:200000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s4_r.xls');
writematrix(real(H_s(200001:250000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s5_r.xls');

writematrix(imag(H_s(1:50000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s1_i.xls');
writematrix(imag(H_s(50001:100000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s2_i.xls');
writematrix(imag(H_s(100001:150000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s3_i.xls');
writematrix(imag(H_s(150001:200000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s4_i.xls');
writematrix(imag(H_s(200001:250000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s5_i.xls');

writematrix(real(H_p(1:50000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p1_r.xls');
writematrix(real(H_p(50001:100000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p2_r.xls');
writematrix(real(H_p(100001:150000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p3_r.xls');
writematrix(real(H_p(150001:200000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p4_r.xls');
writematrix(real(H_p(200001:250000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p5_r.xls');

writematrix(imag(H_p(1:50000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p1_i.xls');
writematrix(imag(H_p(50001:100000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p2_i.xls');
writematrix(imag(H_p(100001:150000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p3_i.xls');
writematrix(imag(H_p(150001:200000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p4_i.xls');
writematrix(imag(H_p(200001:250000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p5_i.xls');

%信道重建数据集
writematrix(real(H(1:50000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_1_r.xls');
writematrix(real(H(50001:100000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_2_r.xls');
writematrix(real(H(100001:150000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_3_r.xls');
writematrix(real(H(150001:200000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_4_r.xls');

writematrix(imag(H(1:50000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_1_i.xls');
writematrix(imag(H(50001:100000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_2_i.xls');
writematrix(imag(H(100001:150000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_3_i.xls');
writematrix(imag(H(150001:200000,:)), 'E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_4_i.xls');