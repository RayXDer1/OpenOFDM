close all;
clear;
clc;
% warning off;

%% ******************************参数配置******************************
mode          = 15;          %模式0
PSDU_PB       = 16;          %默认16字节   

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
snr_min_value   = 40;
snr_max_value   = 40;
delta_snr       = 1;                            %SNR步长
snr             = snr_min_value : delta_snr : snr_max_value;
total_number    = 1000;                            %每个SNR上的仿真次数
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
err_PSDU        = zeros(1,N);


%将程序所在路径导入中
addpath OFDM_TX\
addpath OFDM_RX\
addpath DATA\


%% ******************************仿真开始******************************
for n  = 1 : N
    for k = 1 : total_number

    %% =========================发送端基带处理=========================
    % ----------------------------信号产生-------------------------------
    [PSDU_PB,PSDU_BPC,PSDU_rate] = mode_control(mode);
    PSDU_data_orgi = data_gen(PSDU_PB); %产生随机码元数据

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
    % 块状导频 暂时未做 用前导代替
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

    %% =========================接收端基带处理=========================
    % ----------------------------同步-----------------------------------
    [LOC_LTF] = ltf_gen(CarrierNUM,N_FFT);  %根据可用子载波数生成前导
    % 共有4个LTF 将连续的两个LTF视为一个LTF 然后进行64点互相关运算 可以提高在低信噪比下的同步性能
    LOC_L = LOC_LTF(1,1:1*N_FFT); %取出单个不重复的LTF

    % load("rx_signal.mat");
    w = 8;  %阈值加权因子
    peak_index = symbol_synchron(rx_signal,LOC_L,w); %同步位置
    % peak_index = 201;

    peak_index_L = peak_index(1);

    rx_synchron_signal = rx_signal(1,peak_index_L:end);%同步到的OFDM信号
    rx_preamble_signal = rx_signal(1,peak_index_L:peak_index_L+4*N_FFT-1);%前导OFDM信号

    % 统计同步位置偏差和同步概率
    MSE_tongbu = abs(peak_index_L(1) - 201);
    synchron_local(n) = synchron_local(n) + MSE_tongbu;
    if MSE_tongbu < 32
        synchron_proba(n) = synchron_proba(n) + 1;
    end

    % ----------------------------频偏校正-------------------------------
    [cfo_estimate_value,rx_compen_signal] = cfo_estimate(rx_synchron_signal,rx_preamble_signal,Fs,N_FFT);
    MSE_cfo(n) = MSE_cfo(n) + abs(cfo_estimate_value - cfo_offset); 

    %取消注释则不经过频偏校正
    % rx_compen_signal = rx_synchron_signal;
    
    N_LTF = 4.5 * N_FFT; %计算前导长度
    N_PSDU = Nsym_CP * symbol_num; %计算OFDM的PSDU数据长度

    %输出数据
    rx_ofdm_cpdata = rx_compen_signal(1,N_LTF+1:N_LTF+N_PSDU);
    rx_preamble_data = rx_compen_signal(1,1:4*N_FFT);

    % ----------------------------去循环前缀-----------------------------
    [rx_ofdm_data] = remove_cp(rx_ofdm_cpdata,symbol_num,N_FFT,N_CP);

    % ----------------------------FFT变换--------------------------------
    [fft_out] = fft_trans(rx_ofdm_data,N_FFT);

    %绘制完整的收端星座图
    % scatterplot(fft_out(:)); axis([-2 2 -2 2]);

    % ----------------------------导频提取-------------------------------
    [rx_data,rx_pilot,curr_pilot] = pilot_extract(fft_out,symbol_num,Pilot_method,VCNum_method,Np,N_FFT,ValidCarrierNum,phase);

    %绘制完整的PSDU数据星座图
    % scatterplot(rx_data(:)); axis([-2 2 -2 2]);

    % ----------------------------信道估计与均衡-------------------------
    % 先通过LS估计出信道值
    [H_LS_EST,H_LS_EST_CAR] = channel_estimate(rx_pilot,loc_pilot,symbol_num,Np,N_FFT,Pilot_method,curr_pilot,ValidCarrierNum);

    % H_LS_EST_CAR_TRS = H_LS_EST_CAR.';
    % Hls_est_power_dB = 10*log10(abs(1./H_LS_EST_CAR_TRS.*conj(1./H_LS_EST_CAR_TRS)));
    % figure;plot((Hls_est_power_dB),'-o','LineWidth',2);title("LS信道估计-db");

    % 绘制信道估计的实部、虚部图
    % H_LS_EST_CAR_TRS = H_LS_EST_CAR.';
    % figure;
    % subplot(211);plot(real(H_LS_EST_CAR_TRS(:)));
    % subplot(212);plot(imag(H_LS_EST_CAR_TRS(:)));

    % 再进行信道均衡
    [RX_EQ_OUT] = channel_equalize(rx_data,H_LS_EST);

    % 取消注释则不做信道均衡
    % RX_EQ_OUT = rx_data;

    %绘制均衡前后的PSDU数据星座图
    % scatterplot(rx_data(:)); axis([-2 2 -2 2]);
    % scatterplot(RX_EQ_OUT(:)); axis([-2 2 -2 2]);

    % ----------------------------星座解映射-----------------------------
    %暂时做硬判决
    [DEMOD_OUT] = De_Mod_Map(PSDU_BPC,RX_EQ_OUT,wlanSymMap);

    % ----------------------------去除填充-------------------------------
    [remove_paddata_out] = Remove_Padding(DEMOD_OUT,PadBits_Num); %交织完后填充符号数据

    % ----------------------------解交织---------------------------------
    % 目前未做解交织，之后补上
    PSDU_deinter_out = remove_paddata_out;
    PSDU_deinter_compare = reshape(PSDU_deinter_out,length(PSDU_deinter_out)/2,2);

    % ----------------------------译码-----------------------------------
    % 目前未做译码，之后补上
    PSDU_decode_out = PSDU_deinter_out(1:PSDU_PB * 8);

    % ----------------------------统计-----------------------------------
    [errNum_PSDU,err_PSDU_n] = biterr(PSDU_decode_out,PSDU_data_orgi);
    err_PSDU(n) = err_PSDU(n) + err_PSDU_n;

    % 计数自增
    frame_cnt = frame_cnt + 1

    end
end

% 统计错误率
synchron_local = synchron_local ./ total_number;
synchron_proba = synchron_proba ./ total_number;
MSE_cfo        = MSE_cfo ./ total_number;
err_PSDU       = err_PSDU ./ total_number;


%% ******************************打印输出******************************
% 同步错误率
figure;semilogy(snr, synchron_local,'-ko','LineWidth',2);grid on;
xlabel('snr/dB'), ylabel('同步偏差值'); title('符号同步绝对值偏差曲线');

figure;semilogy(snr, synchron_proba,'-ko','LineWidth',2);grid on;
xlabel('snr/dB'), ylabel('同步捕获概率'); title('符号同步捕获概率曲线');

%CFO估计偏差
figure;semilogy(snr, MSE_cfo,'-ko');grid on;
xlabel('snr/dB'), ylabel('delta f'); title('CFO 估计差值');

%误码率
figure;semilogy(snr,(err_PSDU),'-bo');grid on;
xlabel('信噪比snr/dB');ylabel('PSDU误码率BER');


