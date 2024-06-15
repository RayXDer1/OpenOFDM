close all;
clear;
clc;
% warning off;

%% ******************************参数配置******************************
mode          = 4;           %模式0
PSDU_PB       = 16;           %默认16字节   

% PSDU的MCS设置： 调制方式、码率、数据速率(Mb/s)、符号数、填充数、PB大小
%0  0000    BPSK    0.5 2.25 XXX XXX 16 
%1  0001    BPSK    0.8 3.60 XXX XXX 16 
%2  0010    QPSK    0.5 4.50 XXX XXX 16 
%3  0011    QPSK    0.8 7.20 XXX XXX 16 
%4  0100    16QAM   0.5 9.00 XXX XXX 16 
%5  0101    16QAM   0.8 14.4 XXX XXX 16 
%6  0110    16QAM   0.5 9.00 XXX XXX 72 
%7  0111    16QAM   0.8 14.4 XXX XXX 72 
%8  1000    16QAM   0.5 9.00 XXX XXX 136
%9  1001    16QAM   0.8 14.4 XXX XXX 136
%10 1010    BPSK    0.5 2.25 XXX XXX 520
%11 1011    QPSK    0.5 4.50 XXX XXX 520
%12 1100    16QAM   0.5 9.00 XXX XXX 520
%13 1101    64QAM   0.5 13.5 XXX XXX 520
%14 1110    256QAM  0.5 18.0 XXX XXX 520
%15 1111    1024QAM 0.5 22.5 XXX XXX 520

%16 10000   BPSK    0.5 22.5 XXX XXX 4

%% 测试仿真参数
% 仿真参数
snr_min_value   = -4;
snr_max_value   = 30;
delta_snr       = 1;                            %SNR步长
snr             = snr_min_value : delta_snr : snr_max_value;
total_number    = 2000;                           %每个SNR上的仿真次数
N               = length(snr);
total_numbers   = total_number * N;             %仿真总次数
cfo_offset      = 100;                          %频偏参数：Hz  估计范围：正负156.25KHz

% OFDM参数
N_FFT           = 32;                           %FFT/IFFT点数
VCNum_Opt       = [18 24];                      %有效子载波数目选项(2种数据传输方式,默认18)
VCNum_method    = 1;                            %数据子载波数目：0-18个子载波 1-24个子载波
ValidCarrierNum = VCNum_Opt(VCNum_method+1);    %有效子载波数目
SubcarrierSpace = 312.5e3;                      %子载波间隔 固定为312.5KHz
Fs              = N_FFT*SubcarrierSpace;        %基带采样率 10MHz 带宽为10MHz
N_CP            = N_FFT / 4;                    %保护间隔长度
Nsym_CP         = N_FFT + N_CP;                 %OFDM符号点数：40
Pilot_method    = 3;                            %导频插入方式：0-块状导频 1-梳状导频:2P 2-梳状导频:4P 3-梳状导频:6P
Np_Opt          = [ValidCarrierNum 2 4 6];      %导频数选项
Np              = Np_Opt(Pilot_method+1);       %导频数
CarrierNUM_Opt  = [ValidCarrierNum ValidCarrierNum+2 ValidCarrierNum+4 ValidCarrierNum+6];  %可用子载波数目选项
CarrierNUM      = CarrierNUM_Opt(Pilot_method+1);                                           %可用子载波数目
T_Ofdm          = N_FFT/Fs;                     %秒 一个OFDM符号有效数据持续的时间
Tsym_Ofdm       = Nsym_CP/Fs;                   %秒 一个OFDM符号持续的时间

% 初始化参数
frame_cnt       = 1;                            %帧计数
synchron_local  = zeros(1,N);                   %同步位置偏差
synchron_proba  = zeros(1,N);                   %同步概率偏差

MSE_cfo         = zeros(1,N);
MSE_CNN         = zeros(1,N);
MSE_RESCNN      = zeros(1,N);
MSE_LS          = zeros(1,N);
err_PSDU        = zeros(1,N);
err_DL_PSDU     = zeros(1,N);

Training_set_ratio = 0.5;
Train_data_cnn = zeros(2, 30, 1, Training_set_ratio * total_numbers);
Train_label_cnn = zeros(2, 30, 1, Training_set_ratio * total_numbers);
Valid_data_cnn = zeros(2, 30, 1, total_numbers - Training_set_ratio * total_numbers);
Valid_label_cnn = zeros(2, 30, 1, total_numbers - Training_set_ratio * total_numbers);


%将程序所在路径导入中
addpath OFDM_TX\
addpath OFDM_RX\
addpath DATA\
addpath DEEP_LEARN2\

load("Train_out_cnn.mat");
load("Train_Info_cnn.mat");

load("Train_out_rescnn.mat");
load("Train_Info_rescnn.mat");


%% ******************************仿真开始******************************
for n  = 1 : N
    for k = 1 : total_number

    %% =========================发送端基带处理=========================
    % ----------------------------信号产生-------------------------------
    % mode = randi(16) - 1;

    [PSDU_PB,PSDU_BPC,PSDU_rate] = mode_control(mode);
    PSDU_data_orgi = data_gen(PSDU_PB); %产生随机码元数据
    % PSDU_data_orgi = zeros(1,PSDU_PB*8);

    % 计算当前模式的数据速率 数据速率 = [(每个子载波携带的比特数量 * 编码效率 * 数据子载波数量 / ofdm符号长度) / 1e6] Mb/s
    data_bps = PSDU_BPC * PSDU_rate * ValidCarrierNum / Tsym_Ofdm / 1e6;
    
    % ----------------------------编码-----------------------------------
    % 目前未做编码，之后补上
    PSDU_complen = (PSDU_PB * 8 / PSDU_rate - PSDU_PB * 8);
    PSDU_encode_out = [PSDU_data_orgi PSDU_data_orgi(1,1:PSDU_complen)];
    
    % ----------------------------交织-----------------------------------
    % 目前未做交织，之后补上
    PSDU_inter_out = PSDU_encode_out;

    % ----------------------------符号填充-------------------------------
    [pad_data_out,symbol_num,PadBits_Num] = Symbol_Padding(PSDU_inter_out,ValidCarrierNum,PSDU_BPC); %交织完后填充符号数据

    % ----------------------------星座点映射-----------------------------
    [MOD_OUT,wlanSymMap] = Mod_Map(PSDU_BPC,pad_data_out,ValidCarrierNum,symbol_num);
   
    % ----------------------------导频插入-------------------------------
    [Pilot_Insert_out,loc_pilot,phase] = Pilot_Insert(MOD_OUT,symbol_num,Pilot_method,VCNum_method,Np,N_FFT,ValidCarrierNum);

    % 绘制完整的发端星座图
    % scatterplot(Pilot_Insert_out(:)); axis([-2 2 -2 2]);

    % ----------------------------IFFT变换-------------------------------
    [ifft_out] = ifft_trans(Pilot_Insert_out,N_FFT);

    % ----------------------------循环前缀&加窗--------------------------
    % 目前未做加窗，之后补上
    [CP_ADD_DATA] = add_cp(ifft_out,symbol_num,N_FFT,N_CP);

    % 绘制OFDM频谱图
    % figure;pwelch(CP_ADD_DATA(:),[],[],[],Fs,'centered','psd');
    % title('CP\_ADD\_DATA 10MHz sample power spectrum1');

    % ----------------------------组帧-----------------------------------
    [tx_signal_out] = combine_frame(CP_ADD_DATA,CarrierNUM,N_FFT);

    % 绘制OFDM数据时域图
    % figure;
    % subplot(211);plot(real(tx_signal_out));
    % subplot(212);plot(imag(tx_signal_out));

    % 绘制OFDM频谱图
    % figure;pwelch(tx_signal_out(:),[],[],[],Fs,'centered','psd');
    % title('tx\_signal\_out 10MHz sample power spectrum2');

    %% =========================加噪声、加频偏=========================
    [noise_signal_out,perfect_signal_out,h_out] = add_channel(tx_signal_out,snr(n),cfo_offset,Fs,frame_cnt);
    
    % h_out_trs =  h_out.';
    % H_real_power_dB = 10*log10(abs(1./h_out_trs.*conj(1./h_out_trs)));
    % figure;plot((H_real_power_dB),'-o','LineWidth',2);title("真实信道值-db");

    % 绘制加噪后的OFDM频谱图
    % figure;pwelch(noise_signal_out(:),[],[],[],Fs,'centered','psd');
    % title('noise\_signal\_out 10MHz sample power spectrum3');

    % 绘制加噪后的OFDM数据时域图
    % figure;
    % subplot(211);plot(real(noise_signal_out));
    % subplot(212);plot(imag(noise_signal_out));

    % 数据接收
    rx_signal = noise_signal_out;
    rx_perfect_signal = perfect_signal_out;

    %% =========================接收端基带处理=========================

    %在时域上采集含(不含)噪声的OFDM数据
    [HPREAMBLE_LS,HPREAMBLE_PLS,rx_ofdm_data,rx_pofdm_data,synchron_local,synchron_proba,MSE_cfo] = rx_lscnn_test1(synchron_local,synchron_proba,MSE_cfo,rx_signal,rx_perfect_signal,CarrierNUM,symbol_num,N_FFT,N_CP,Nsym_CP,cfo_offset,Fs,n);
    % [PSDU_decode_out] = rx_lscnn_test2(rx_ofdm_data,symbol_num,Pilot_method,VCNum_method,Np,loc_pilot,phase,N_FFT,ValidCarrierNum,PSDU_BPC,wlanSymMap,PadBits_Num,PSDU_PB);

    % HPREAMBLE_PLS_EST = HPREAMBLE_LS - HPREAMBLE_PLS;
    HPREAMBLE_PLS_EST = HPREAMBLE_PLS;

    % 采集数据
    % [Train_data_cnn,Train_label_cnn,Valid_data_cnn,Valid_label_cnn] = dl_cnn_data_collect(Train_data_cnn,Train_label_cnn,Valid_data_cnn,Valid_label_cnn,rx_ofdm_data,rx_nofdm_data,frame_cnt,total_numbers,Training_set_ratio);
    [data,label] = dl_lscnn_data_collect(HPREAMBLE_LS,HPREAMBLE_PLS_EST);

    if frame_cnt <= fix(Training_set_ratio * total_numbers)
        Train_data_cnn(:, :, :, frame_cnt) = data;
        Train_label_cnn(:, :, :, frame_cnt) = label;
    else
        Valid_data_cnn(:, :, :, frame_cnt - Training_set_ratio * total_numbers) = data;
        Valid_label_cnn(:, :, :, frame_cnt - Training_set_ratio * total_numbers) = label;
    end

    % 完美去噪后的信号解调
    % [PSDU_pdecode_out] = rx_lscnn_test2(rx_pofdm_data,symbol_num,Pilot_method,VCNum_method,Np,loc_pilot,phase,N_FFT,ValidCarrierNum,PSDU_BPC,wlanSymMap,PadBits_Num,PSDU_PB);

    %% 深度学习时域降噪方法
    % 深度学习降噪后的信号解调
    HPREAMBLE_PLS_CNNOUT = HPREAMBLE_LS;
    CNN_input_data = [real(HPREAMBLE_LS);imag(HPREAMBLE_LS)];
    CNN_all = CNN_input_data(:);
    CNN_alls = 2*(CNN_all - min(CNN_all))./(max(CNN_all)-min(CNN_all))-1;
    reshape_CNN_input_data = reshape(CNN_alls,2,30);
    Received_data_CNN = predict(Train_out_cnn,reshape_CNN_input_data);
    Received_data_CNN_all = Received_data_CNN(:);
    pre_Received_data_CNN = (Received_data_CNN_all + 1).*(max(CNN_all)-min(CNN_all))./2 + min(CNN_all);
    preshape_Received_data_CNN = reshape(pre_Received_data_CNN,2,30);
    HPREAMBLE_PLS_CNNOUT = preshape_Received_data_CNN(1,:) + preshape_Received_data_CNN(2,:) * 1i;

    Received_data_RESCNN = predict(Train_out_rescnn,reshape_CNN_input_data);
    Received_data_RESCNN_all = Received_data_RESCNN(:);
    pre_Received_data_RESCNN = (Received_data_RESCNN_all + 1).*(max(CNN_all)-min(CNN_all))./2 + min(CNN_all);
    preshape_Received_data_RESCNN = reshape(pre_Received_data_RESCNN,2,30);
    HPREAMBLE_PLS_RESCNNOUT = preshape_Received_data_RESCNN(1,:) + preshape_Received_data_RESCNN(2,:) * 1i;

    MSE_CNN0 = (HPREAMBLE_PLS_CNNOUT - HPREAMBLE_PLS) * (HPREAMBLE_PLS_CNNOUT - HPREAMBLE_PLS)' ./ (length(HPREAMBLE_PLS));
    MSE_RESCNN0 = (HPREAMBLE_PLS_RESCNNOUT - HPREAMBLE_PLS) * (HPREAMBLE_PLS_RESCNNOUT - HPREAMBLE_PLS)' ./ (length(HPREAMBLE_PLS));
    MSE_LS0 = (HPREAMBLE_LS - HPREAMBLE_PLS) * (HPREAMBLE_LS - HPREAMBLE_PLS)' ./ (length(HPREAMBLE_PLS));
    MSE_CNN(n) = MSE_CNN(n) + MSE_CNN0;
    MSE_RESCNN(n) = MSE_RESCNN(n) + MSE_RESCNN0;
    MSE_LS(n) = MSE_LS(n) + MSE_LS0;

    % 深度学习方法
    % [PSDU_DL_decode_out] = rx_cnn_test2(rx_cnn_ofdm_data,symbol_num,Pilot_method,VCNum_method,Np,loc_pilot,phase,N_FFT,ValidCarrierNum,PSDU_BPC,wlanSymMap,PadBits_Num,PSDU_PB);

    % ----------------------------统计-----------------------------------
    % [errNum_PSDU,err_PSDU_n] = biterr(PSDU_decode_out,PSDU_data_orgi);
    % err_PSDU(n) = err_PSDU(n) + err_PSDU_n;
    % 
    % [errNum_PSDU_DL,err_PSDU_DL_n] = biterr(PSDU_DL_decode_out,PSDU_data_orgi);
    % err_DL_PSDU(n) = err_DL_PSDU(n) + err_PSDU_DL_n;

    % 计数自增
    frame_cnt = frame_cnt + 1

    end
end

% 统计错误率
synchron_local = synchron_local ./ total_number;
synchron_proba = synchron_proba ./ total_number;
MSE_cfo        = MSE_cfo ./ total_number;
MSE_CNN        = MSE_CNN ./ total_number;
MSE_RESCNN     = MSE_RESCNN ./ total_number;
MSE_LS         = MSE_LS ./ total_number;
err_PSDU       = err_PSDU ./ total_number;
err_DL_PSDU    = err_DL_PSDU ./ total_number;

% 采集完数据 开始CNN训练
% [Train_out_cnn, Train_Info_cnn] = Train_LSCNN_V1(Train_data_cnn, Train_label_cnn, Valid_data_cnn, Valid_label_cnn);


%% ******************************打印输出******************************
% 同步错误率
figure;semilogy(snr, synchron_local,'-ko','LineWidth',2);grid on;
xlabel('snr/dB'), ylabel('同步偏差值'); title('符号同步绝对值偏差曲线');

figure;semilogy(snr, synchron_proba,'-ko','LineWidth',2);grid on;
xlabel('snr/dB'), ylabel('同步捕获概率'); title('符号同步捕获概率曲线');

% CFO估计偏差
figure;semilogy(snr, MSE_cfo,'-ko');grid on;
xlabel('snr/dB'), ylabel('delta f'); title('CFO 估计差值');

% 前导处的信道估计偏差
figure;semilogy(snr,(MSE_LS),'-bo','LineWidth',2);grid on;
hold on;semilogy(snr,(MSE_CNN),'-ro','LineWidth',2);grid on;
semilogy(snr,(MSE_RESCNN),'-go','LineWidth',2);grid on;
title("信道CSI估计MSE曲线图");
xlabel('信噪比snr/dB');ylabel('MSE');
legend("LS","LSCNN","LSRESCNN");
hold off;

% 误码率
figure;semilogy(snr,(err_PSDU),'-bo','LineWidth',2);grid on;
hold on;
semilogy(snr,(err_DL_PSDU),'-ro','LineWidth',2);
legend("未降噪","TD\_CNN");
xlabel('信噪比snr/dB');ylabel('PSDU误码率BER');
hold off;

