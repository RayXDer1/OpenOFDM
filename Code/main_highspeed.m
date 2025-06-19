clear;
close all;
warning off;

%% ---------------参数配置---------------------------------------------------
%% 信号参数配置

%PHR的MCS设置： 调制方式、码率、数据速率(kb/s)、符号数、填充数、PB大小
%0    0 0 0      BPSK     0.5      3.6625       15      14      1       opt1超短帧模式
%1    0 0 1      BPSK     0.8      5.8625       9       2       1      	opt1超短帧模式
%2    0 1 0      BPSK     0.5      58.6         15      14      16     	opt2短帧模式
%3    0 1 1      BPSK     0.8      93.8         9       2       16     	opt2短帧模式
%4    1 0 0      QPSK     0.5      117.2        8       32      16     	opt2短帧模式
%5    1 0 1      QPSK     0.8      187.5        5       20      16   	opt2短帧模式
%6    1 1 0      BPSK     0.5      58.6         66      0       72     	opt3长帧模式
%7    1 1 1      BPSK     0.5      58.6         237     6       264     opt3长帧模式 

%--------------------------禁用---------------------------
%6    1 1 0      16QAM    0.5      234.4        4       32      
%7    1 1 1      16QAM    0.8      375.0        3       56   

global Opt;
Opt = 1;
SIG_data = [0,0,0];         %3bit   opt1支持000 001 opt2支持010 011 100 101 opt3支持110 111
PRDU_pbsize = 1;            %1字节  opt1的PRDU的数据长度为PB1,Turbo编码块长度为PB16
PHR_PBSIZE = [16 72 264];   %16字节 opt2的PHR的Turbo编码块长度为PB16 opt3的PHR的Turbo编码块长度为PB72、PB264 
 
%% 测试常变的参数
minvaule = -8;             %0
maxvaule = 5;              %5
deltaSNR = 1;               %步长
snr = minvaule : deltaSNR : maxvaule;   %信噪比范围（dB）
total_number = 2000;          %每次发送帧的个数1e1
zhen = 0;                   %帧计数
write = 0;                  %为1代表量化后输出到txt文件

%% 基本不变的参数
iter_num = 8;               %turbo译码迭代次数
Freq_offset = 0;            %载波频偏cfo=Freq_offset/fu=Freq_offset*Tofdm
Lc = 2;                     %信道可信度 2

IFFT_NUM = 32;              %IFFT点数，矩阵
FFT_NUM = 32;               %FFT点数，矩阵
Tofdm = 122.88e-6;          %秒 一个OFDM符号持续的时间
Fs0 = FFT_NUM./Tofdm;       %基本采样速率，数，点数/符号长度 % Fs0 = [0.2604e6];
n_Fs0 = 160;                %一次过采样倍数
Fsn = n_Fs0.*Fs0;           %n倍采样速率[41.6667]M
n_Fs4 = 10;                 %一次下采样倍数
Fs4 = Fsn./n_Fs4;           %一次下采样后的采样速率，实际是4.16667M
n_Fs4_0 = 16;               %二次下采样倍数
downs = 16;                 %下采样基数
fc = 10e6;                  %Hz 设置载波fc频率
n_Ssymbol = 7;              %不同模式的s符号数不同 用于频偏估计中
q_type_1 = [12,6];          %对数据进行[12,6]量化

%多径时延参数 仿真3径情况
mult_path_amp = [1 0.0 0.0]; %多径幅度 [1 0.2 0.1];
mutt_path_time = [0 10 20];  %多径时延 本系统保护间隔为30.72us 基带速率为1.0416M 单位为0.96us

%% 初始化参数
addpath Reciver\
addpath Spread\
N = length(snr);
err_channel_PHR = zeros(1,N);
err_PHR = zeros(1,N);
err_PRDU = zeros(1,N);
err_SIG = zeros(1,N);
MSE_CFO = zeros(1,N);
index_S = zeros(1,N);
index_L = zeros(1,N);
pick = zeros(1,N);
wr_location = zeros(1,2*N);

%% --------------------仿真开始-------------------------------------
for n  = 1 : N
    for k = 1 : total_number

%% ==============发送端基带处理 ============================================= 
%% ----------------1.信号产生------------------------------
        [PHR_CopyNum,PHR_BPC,PHR_rate] = sig_control(SIG_data);
        [PHR_out,PHR_data_orgi,PRDU_orgi,PHR_pbsize] = data_produce(SIG_data,PHR_PBSIZE,PRDU_pbsize);   
        PHR_scramb_out = scramb_code(PHR_out);
        
%% ----------------2.1扩频-------------------
        [PRDU_data,M] = DSSS_Spread(PRDU_orgi,PRDU_pbsize*8);
        repeat_PRDU = repelem(PRDU_orgi,16); %每个元素重复16次
        
        % 根据SIG选择重传帧还是数据帧
        if isequal(SIG_data,[0,0,0]) || isequal(SIG_data,[0,0,1])
            PHR_data = repeat_PRDU;
        else
            PHR_data = PHR_scramb_out;
        end
        
        % figure;
        % plot(PRDU_orgi);
        % figure;
        % plot(PRDU_data);
        
%% ----------------2.2SIG-------------------
        [SIG_out,SIG_carr_num,SIG_num] = SIG_encoder(SIG_data);

%% ----------------2.3PHR-------------------
% --测试--
    %[PHR_data_orgi,PHR_data] = byte_converge(); %将字节转化为bit
%     [PHR_data_orgi,PHR_data] = byte_converge(); %将字节转化为bit
%     PHR_data = scramb_code(PHR_data);
        %PHR Turbo编码
        PHR_turbo1 = turbo_encoder(PHR_data, PHR_pbsize, PHR_rate);
        PHR_turbo2 = [PRDU_data,PHR_turbo1((PHR_pbsize*8+1):end)];
%         PHR_turbo_out = [PHR_data,PHR_data];   %取消turbo编码
        if isequal(SIG_data,[0,0,0]) || isequal(SIG_data,[0,0,1])
            PHR_turbo_out = PHR_turbo2;
        else
            PHR_turbo_out = PHR_turbo1;
        end 
    
        %PHR 信道交织
        PHR_inter_out = channal_inter(PHR_pbsize,PHR_turbo_out,PHR_rate);       
        %PHR 比特填充  
        [PHR_diver_out,PHR_num,PHR_carr_num,PadBitsNum_PHR] = Bit_Padding(PHR_inter_out,PHR_CopyNum,PHR_BPC);
        

%% ----------------3.星座点映射--------------------------
        %SIG 星座点映射  % SIG 只使用BPSK
        [SIG_IQ_out,SIG_data_phase] = IQ_modulation(1,SIG_out,SIG_carr_num,SIG_num); %SIG只用BPSK调制
        %PHR 星座点映射
        [PHR_IQ_out,PHR_data_phase] = IQ_modulation(PHR_BPC,PHR_diver_out,PHR_carr_num,PHR_num);
        
        %对相位数据进行[16,13]量化 13位小数
        q_type_2 = [16,13];
        [SIG_data_phases] = quanti_txt2(SIG_data_phase,q_type_2,0);     %量化并用于后续计算
        
        %SIG PHR PSDU 数据合并
        IQ_OUT = [SIG_IQ_out;PHR_IQ_out];
        symbol_num = SIG_num + PHR_num;

%% ----------------4.插入导频---------------------------------  
        [Pilot_OUT,scramble_patt] = ADD_Pilot(symbol_num,IQ_OUT);
        
        %对导频数据进行[8,6]量化 6位小数
        q_type_3 = [8,6];
        [scramble_patts] = quanti_txt(scramble_patt,q_type_3,write);
        
        %% 相位旋转
        %对插入导频后的数据进行相位旋转，相位载波号如下：
        % data_phase = SIG_data_phases;    %两个相位表一样,任选一个即可 -16到16
        % [Phase_Rotate_Out] = phase_rotate(data_phase,Pilot_OUT,symbol_num,FFT_NUM);
        
%% ----------------5.IFFT-----------------------------
        % IFFT_OUT = ifft_transform(symbol_num,Phase_Rotate_Out,IFFT_NUM);
        IFFT_OUT = ifft_transform(symbol_num,Pilot_OUT,IFFT_NUM);
%% ----------------6.循环前缀&加窗---------------------
        [CP_ADD] = add_cyclic_prefix(IFFT_OUT.',symbol_num);

%% ----------------7.组帧-----------------------------
        %加前导，组帧
        [Send_Signal] = make_frame(CP_ADD);  %ofdm功率没有统一 
%         [pxx,f] = pwelch(Send_Signal,Fs0);
%         figure;pwelch(f,10*log10(pxx))
%         xlabel('Frequency (Hz)')
%         ylabel('PSD (dB/Hz)')
%% ----------------8.成型滤波_v2-----------------------------
%--测试--
        Fd = 0.26041666; 
        Fs = Fd*4;
        Delay = 3;  %阶数=Delay * Fs/Fd * 2 + 1 = size(yf)
        R = 0.35; 
%         [yf,tf] = rcosine(Fd,Fs,'fir/sqrt',R,Delay);
%         [yf_rx,tf_rx] = rcosine(Fd,Fs*4,'fir/sqrt',R,Delay);
        sps = Fs/(Fd);span = 24/sps;       
        yf = rcosdesign(R,span,sps,'sqrt')*2; %乘以以确保幅值统一
        tf = 1;
        sps_rx = 4*Fs/(Fd);span_rx = 128/sps_rx;       
        yf_rx = rcosdesign(R,span_rx,sps_rx,'sqrt');
        tf_rx = 1;
        [yf] = quanti_txt(yf,[12 10],1);%量化并用于后续计算
        [yf_rx] = quanti_txt(yf_rx,[12 10],1);%量化并用于后续计算
        

%         Send_SignalCopy = upsampleCopy(Send_Signal,Fs/Fd); %copy4 发端需要复制点数 在1.041666MHz上发送
        Send_SignalCopy = zeros(1,length(Send_Signal)*4);
        Send_SignalCopy(1:4:end) = Send_Signal; %%插值后信号
        Send_SignalCopys = Send_SignalCopy ;
        
%         figure(1);
%         plot(yf); grid;title('根升余弦滤波器时域波形')
        xt_r = real(Send_SignalCopys);
        xt_i = imag(Send_SignalCopys);  
        yr = filter(yf,tf,xt_r); %%成型滤波 
        yi = filter(yf,tf,xt_i); %%成型滤波        
        Send_Signal_shape = yr + yi * 1i;
        
        %施加两径瑞利信道
        [Send_Signals,H] = add_channel(Send_Signal_shape,0);
        
%% ==============收发射频信号处理 ===========================================        

%% -------------1.上采样 DA变换---------------------------
        Send_Signal_Fsn = interp(Send_Signals,40);%方法6——时域处理

%% -------------2.上变频----------------------------
        Send_Signal_Fsn_fc = UpConversion(Send_Signal_Fsn,fc,Fs0*160);%输入参数1 上采样后的信号，2 倍频的载波频率
        
        Send_Signal_Fsn_fc1 = fft(Send_Signal_Fsn_fc);
        L = length(Send_Signal_Fsn_fc);
        

%% -------------3.仿真信道噪声和频偏----------------------------
        [Send_Signal_Channel,Send_Signal_Fsn_fc,snr_out] = Channel(Send_Signal_Fsn_fc,snr(n),Freq_offset,Fs0*160);
        
%% -------------4.下变频--------------------------------
        Re_Signal_Fsn_defc = DownConversion(Send_Signal_Channel,fc,Fs0,160);

%% -------------5.下采样 AD变换 至原数据速率的4倍，即n/4------------------
        Re_Signal_Fs40 = downsample(Re_Signal_Fsn_defc,n_Fs4*4); %n_Fs0 滤波器
        %仿真多径时延
        path2 = mult_path_amp(2)*[zeros(1,mutt_path_time(2)) Re_Signal_Fs40(1:end-mutt_path_time(2))];
        path3 = mult_path_amp(3)*[zeros(1,mutt_path_time(3)) Re_Signal_Fs40(1:end-mutt_path_time(3))];
        Re_Signal_Fs41 = Re_Signal_Fs40 + path2 + path3; % 多径信号
        
%         Re_Signal_Fs42 = upsampleCopy(Re_Signal_Fs41,4);  %接收端4倍过采样
        Re_Signal_Fs42 = zeros(1,length(Re_Signal_Fs41)*4);
        Re_Signal_Fs42(1:4:end) = Re_Signal_Fs41; %%插值后信号
        
%         p1 = Re_Signal_Fs42(10000+k:end);       %在帧前加空隙100
        p1 = zeros(1,1000);
        p2 = zeros(1,3000);      %在帧后加空隙1000
        Re_Signal_Fs43 = [p1,Re_Signal_Fs42,p2];   %前后各补零100,1000
        
        %不经过射频信号,高斯白噪声信道
%         [Re_Signal_Fs53,snr_out] = add_noise(Send_Signals,snr(n),Freq_offset,Fsn);

%% ==============接收端基带处理 ===========================================
%% -------------1.匹配滤波------------------------------------------------------
%--测试-- 测试阶段
        Re_Signal_Fs53 = Re_Signal_Fs43;
        yr1 = filter(yf_rx,tf_rx,real(Re_Signal_Fs53)); %%用与发送端相同的根升余弦匹配滤波
        yi1 = filter(yf_rx,tf_rx,imag(Re_Signal_Fs53)); %%用与发送端相同的根升余弦匹配滤波 
        Re_Signal_Fs54 = (yr1 + yi1 * 1i);

% %          对上版数据进行验证
%         load("din_r1.mat"); %从本地加载
%         load("din_i1.mat"); %从本地加载
% %         
%         [din_r1,din_i1] = ilachange;
%         din_r1 = din_r1/64;
%         din_i1 = din_i1/64;    
%         Re_Signal_Fs55 = din_r1 + din_i1*1i;
%         signal_power = var(Re_Signal_Fs55(1,8215:20406));
%         noise_power = var(Re_Signal_Fs55(1,20406:32597));
%         t =  signal_power/noise_power;
%         
%         save("Re_Signal_Fs54.mat","Re_Signal_Fs54");
%         load("Re_Signal_Fs54.mat"); %从本地加载 
%        [Re_Signal_Fs54] = quanti_txt(Re_Signal_Fs54,q_type_1,write);%量化并用于后续计算  
        
        data_syn_S = Re_Signal_Fs54;  %接收数据,进行后续的基带处理     1.3    11
        [data_syn_S] = quanti_txt(data_syn_S,q_type_1,write);%量化并用于后续计算  
 
%         data_syn_SF = fft(data_syn_S,512);
%% -------------4.细同步----------------------------------------------
        %----------L符号产生--------------------------------------------
        loc_LTF = LTFGen();
        N_loc_L = length(loc_LTF)/2.5; %本地L符号的长度
        N_loc_L4 = n_Fs4_0*N_loc_L;%4倍采样时，本地L符号的长度
        loc_L = loc_LTF(1,1:N_loc_L); %[GI 0.5个L]
        % loc_L4 = interp(loc_L,4); %4倍采样的本地s符号
        
        % 产生连续的LTF本地序列，能更好地估计信道特性
        loc_L4 = upsampleFreq(loc_L,n_Fs4_0); %4倍采样的本地s符号
        [loc_L4] = quanti_txt(loc_L4,q_type_1,write);%量化并用于后续计算

        % 产生复制的LTF本地序列，能更好地找出细同步准确的位置
        loc_L41 = upsampleCopy(loc_L,n_Fs4_0); %4倍采样的本地s符号
        [loc_L41] = quanti_txt(loc_L41,q_type_1,write);%量化并用于后续计算
        % 量化
%         [data_compen1] = quanti_txt(data_compen1,q_type_1,write);%量化并用于后续计算 
 
        %----------细同步----------------------------------------------  
%         peak_index_L = tongbuL(data_compen1,loc_L41);
%         if k<=total_number/2
%            p = 0;
%            wr = wr_location(n);
%         else
%            p = snr(n); 
%            wr = wr_location(n+N);
%         end
        
%         [peak_index_L,wr_add] = tongbuL_freq(data_syn_S,loc_L4,1,0);
        [peak_index_L,fre] = pmf_fft(data_syn_S,loc_L4);
        data_syn_S=data_syn_S(1,peak_index_L:end);
        [peak_index_L,peak]= tongbuL_new(data_syn_S,loc_L4);
        index_L(n) = peak_index_L(1);
%         index_L(n) = 1278;
        
%         if k<=total_number/2
%            wr_location(n) = wr_add;
%         else 
%            wr_location(n+N) = wr_add;
%         end
        %----------不做细频偏----------------------------------------------
        N_LTF = n_Fs4_0*2.5*FFT_NUM;N_PPDU = n_Fs4_0*FFT_NUM*1.25*symbol_num;%后续不频偏时只用PPDU
        data_offset_L1 = data_syn_S(1,index_L(n):end);
        
%         % 测试根据LTF提取出噪声信号功率
%         pick_n = pick_noise(data_offset_L1,loc_L41,n);
%         pick(n) = pick(n) + pick_n;
        
        [data_offset_L1s] = quanti_txt(data_offset_L1,q_type_1,write);%量化并用于后续计算1..
        data_offset_L2 = data_offset_L1s(1,N_LTF+1:N_LTF+N_PPDU); %特别要注意
        data_pilot2 = data_syn_S(1,index_L(n):index_L(n)+2*N_loc_L4-1); % l l      
%         data_offset_L =data_compen1(1,index_L(n)+N_LTF:index_L(n)+N_LTF+N_PPDU-1);
%         data_pilot2 = data_compen1(1,index_L(n):index_L(n)+2*N_loc_L4-1); % l l

        %----------做细频偏---名字是反的--方便去掉-----------------------------------------
        data_compen2 = data_syn_S(1,index_L(n):end);%后续二次频偏时用的数据
        data_pilot_compen2 = data_compen2(1:2*N_loc_L4); % l l
        [data_compen2] = quanti_txt(data_compen2,q_type_1,write);%量化并用于后续计算  

%% -------------n.LTF 频偏估计----------------------------------------------
        [Fre_estim_vaule2,data_pilot2,data_offset_L2] = freq_offset_L(data_compen2,data_pilot_compen2,Fs4);
        MSE_CFO(n) = MSE_CFO(n)+abs(Freq_offset-Fre_estim_vaule2);  %仅看差值
        %  细频偏补偿后的星座图
%         figure;plot(real(data_offset_L2),imag(data_offset_L2),'.');title("细频偏补偿后星座图");


%         [data_offset_L21] = quanti_txt(data_offset_L2,q_type_1,write);%量化并用于后续计算  
%         fid=fopen('D:\vivado_work\XAXI_AD9361_v16\axi_trx\axi_trx.coe\receiver\write_com\dout_r.txt'); 
%         din_r=textscan(fid,'%f','delimiter','\n'); 
%         fid1=fopen('D:\vivado_work\XAXI_AD9361_v16\axi_trx\axi_trx.coe\receiver\write_com\dout_i.txt'); 
%         din_i=textscan(fid1,'%f','delimiter','\n'); 
%         din_r2 = din_r{1,1}.'/64;
%         din_i2 = din_i{1,1}.'/64;
%         rec_rtl = (din_r2 + din_i2*1i);
%         rec_rtl= rec_rtl(1:24:end); %%抽取后信号
%         data_offset_L22 = data_offset_L21(1:12146);
%         figure;plot(real(data_offset_L22)-real(rec_rtl));title("rec实部差值"); 
%         figure;plot(imag(data_offset_L22)-imag(rec_rtl));title("rec虚部差值"); 
        
%         figure;plot(real(fft_compare(1:8498)));hold on; plot(real(fft_rtl));title("数据对比");


        data_offset_L2 = data_offset_L2(1,N_LTF+1:N_LTF+N_PPDU); %特别要注意
        
       %% -------------SNR信噪比估计-----------------------
        % 测试根据LTF提取出噪声信号功率
        pick_n = pick_noise(data_pilot2,loc_L4,n);
        pick(n) = pick(n) + pick_n;

%% -------------5.去CP---输入data_offset_L倍采样的PPDU数据------------------------------------
        data_cp_removed = remove_cp(data_offset_L2,symbol_num,n_Fs4_0);  

%% -------------6.4倍过采样的FFT------------------------------------------------------------
        fre_data_and_pilot = fft_func(data_cp_removed,n_Fs4_0);
        fre_data_and_pilot32 = fre_data_and_pilot * sqrt(32);
%         fid=fopen('fft_r.txt'); 
%         din_r=textscan(fid,'%f','delimiter','\n'); 
%         fid1=fopen('fft_i.txt'); 
%         din_i=textscan(fid1,'%f','delimiter','\n'); 
%         din_r1 = din_r{1,1}.';
%         din_i1 = din_i{1,1}.';
%         fft_rtl = (din_r1 + din_i1*1i);
%         fft_rtl = fft_rtl(1025:end);
%         fft_compare = reshape(fre_data_and_pilot32,1,8704);
%         figure;plot(real(fft_compare(1:8498))-real(fft_rtl));title("fft实部差值"); 
% %         figure;plot(imag(fft_compare(1:8498))-imag(fft_rtl));title("fft虚部差值"); 
%         figure;plot(real(fft_compare(1:8498)));hold on; plot(real(fft_rtl));title("数据对比");
%% -------------7.信道估计与均衡------------------------------------------------------------
        %-----------信道估计----------------------------------------------
        [chan_estim_vaule] = channel_estim(data_LL_fft,L_fft);
%         chan_estim_vaule32 = chan_estim_vaule./sqrt(32);
%         Hls = ([chan_estim_vaule(2:11) chan_estim_vaule(end-9:end)]);
%         var(real(Hls) - ones(1,20))
%         var(imag(Hls))
%         figure;plot(real(Hls));figure;plot(imag(Hls));

        % -----------信道补偿----------------------------------------------
        [fre_d_and_p_chanCompen4] = channel_compen(fre_data_and_pilot,chan_estim_vaule);
%         fre_d_and_p_chanCompen4 = fre_data_and_pilot; %不进行信道均衡
 
%测试对比数据
%         fid=fopen('D:\vivado_work\XAXI_AD9361_v16\axi_trx\axi_trx.coe\receiver\write_com\dataoutr.txt'); 
%         din_r=textscan(fid,'%f','delimiter','\n'); 
%         fid1=fopen('D:\vivado_work\XAXI_AD9361_v16\axi_trx\axi_trx.coe\receiver\write_com\dataouti.txt'); 
%         din_i=textscan(fid1,'%f','delimiter','\n'); 
%         din_r3 = din_r{1,1}.'/1024;
%         din_i3 = din_i{1,1}.'/1024;
%         xindao_rtl = (din_r3 + din_i3*1i);
%         fre_d_and_p_chanCompen5 = reshape(fre_d_and_p_chanCompen4,1,8704);
%         xindao_rtl = xindao_rtl(1:12146);
%         figure;plot(real(fre_d_and_p_chanCompen5)-real(xindao_rtl));title("rec实部差值"); 
%         figure;plot(imag(fre_d_and_p_chanCompen5)-imag(xindao_rtl));title("rec虚部差值"); 
%         figure;plot(real(fre_d_and_p_chanCompen5));hold on; plot(real(xindao_rtl));title("数据对比");

%--测试--  画出信道均衡前后的星座图进行对比    
        %画出信道均衡前的图
%         figure;plot(real(fre_data_and_pilot(1025:1536)));title("信道均衡前数据实部图"); 
%         figure;plot(imag(fre_data_and_pilot(1025:1536)));title("信道均衡前数据虚部图"); 
%         figure;plot(real(fre_data_and_pilot(:)),imag(fre_data_and_pilot(:)),'.');title("信道均衡前星座图"); 

        %画出信道均衡后的图
%         fre_d_and_p_chanCompen4s = fre_d_and_p_chanCompen4.';
%         real_fre_d_and_p_chanCompen4s = real(fre_d_and_p_chanCompen4s(:));
%         imag_fre_d_and_p_chanCompen4s = imag(fre_d_and_p_chanCompen4s(:));
%         figure;plot(real_fre_d_and_p_chanCompen4s,imag_fre_d_and_p_chanCompen4s,'.');title("信道均衡后星座图"); 

%% -------------8.下采样至原速率---------------------------------------------------------
        fre_d_and_p_chanCompen = fre_d_and_p_chanCompen4([1:downs,(513-downs):512],:)/n_Fs4_0;%n_Fs4_0; %硬件下采样方法
%         fre_d_and_p_chanCompen1 = fre_d_and_p_chanCompen4([(513-downs):512,1:downs],:);%n_Fs4_0; %硬件下采样方法
        %对下采样后的数据进行[12,6]量化
        [fre_d_and_p_chanCompens] = quanti_txt(fre_d_and_p_chanCompen,q_type_1,write);%量化并用于后续计算
        
        %% 相位逆旋转
        % [fre_d_disrotate] = phase_inverse_rotate(fre_d_and_p_chanCompens,data_phase,FFT_NUM,symbol_num,q_type_1,write);
%         fre_d_disrotate1 = fre_d_disrotate * n_Fs4_0;
        
%         fid=fopen('D:\vivado_work\XAXI_AD9361_v16\axi_trx\axi_trx.coe\receiver\write_com\down_doutr.txt'); 
%         din_r=textscan(fid,'%f','delimiter','\n'); 
%         fid1=fopen('D:\vivado_work\XAXI_AD9361_v16\axi_trx\axi_trx.coe\receiver\write_com\down_douti.txt'); 
%         din_i=textscan(fid1,'%f','delimiter','\n'); 
%         din_r4 = din_r{1,1}.'/1024;
%         din_i4 = din_i{1,1}.'/1024;
%         xindao_rtl = (din_r3 + din_i3*1i);
%         fre_d_and_p_chanCompen5 = reshape(fre_d_and_p_chanCompen4,1,8704);
% %         xindao_rtl = xindao_rtl(1:12146);
%         figure;plot(real(fre_d_and_p_chanCompen5)-real(xindao_rtl));title("rec实部差值"); 
%         figure;plot(imag(fre_d_and_p_chanCompen5)-imag(xindao_rtl));title("rec虚部差值"); 
%         figure;plot(real(fre_d_and_p_chanCompen5));hold on; plot(real(xindao_rtl));title("数据对比");        
        
        
%% -------------9.数据拆分----------------------------------------------------------------------
        % [freq_data_SIG,freq_data_PHR,freq_pilot_syms] = DePilot(fre_d_disrotate,SIG_num,PHR_num);
        [freq_data_SIG,freq_data_PHR,freq_pilot_syms] = DePilot(fre_d_and_p_chanCompens,SIG_num,PHR_num);      
%--测试-- 对SIG、PHR数据进行[12,6]量化
        freq_data_SIG = freq_data_SIG.';freq_data_PHR = freq_data_PHR.';
        [freq_data_SIGs] = quanti_txt(freq_data_SIG,q_type_1,write);    %量化SIG并用于后续计算
        [freq_data_PHRs] = quanti_txt(freq_data_PHR,q_type_1,write);    %量化PHR并用于后续计算
        [freq_pilot_symss] = quanti_txt(freq_pilot_syms,q_type_1,write);%量化导频并用于后续计算
        freq_data_SIGs = freq_data_SIGs.';freq_data_PHRs = freq_data_PHRs.';
        
        % 采用导频做SNR估计
%         pick_noise(freq_pilot_symss,scramble_patts,PSDU_num);
        
%% --------------10.相位跟踪-------------------
        % [freq_data_SIG_comp ,freq_data_PHR_comp, phase_comp] = PhaseCorrection(freq_pilot_symss,freq_data_SIGs,freq_data_PHRs,scramble_patts);
        
%% --------------10.去掉相位跟踪-------------------
%--测试-- 取消注释即可去除相位追踪模块，主要验证导频数据是否正确
        freq_data_SIG_comp = freq_data_SIG.';
        freq_data_PHR_comp = freq_data_PHR.';

%% ************************************************************************************************************
%% --------------11.解调-------------------
        SIG_IQ = IQ_demodulation(1,freq_data_SIG_comp,SIG_num);
        PHR_IQ = IQ_demodulation(PHR_BPC,freq_data_PHR_comp,PHR_num);

%--测试-- 画出解调前星座图      
%         figure; plot(real(freq_data_SIG_comp(:)),imag(freq_data_SIG_comp(:)),'.'); title("解调前SIG星座图");
%         figure; plot(real(freq_data_PHR_comp(:)),imag(freq_data_PHR_comp(:)),'.'); title("解调前PHR星座图");

%% --------------12.SIG-------------------
        SIG_bit = decoder_SIG(SIG_IQ); 
        [errNum_SIG,err_SIG_n] = biterr(SIG_bit,SIG_data);
%         if(err_SIG_n > 0)
%             break;
%         end
        err_SIG(n) = err_SIG(n) + err_SIG_n;
        
%% --------------13.PHR------------------
%       PHR 分集合并
        %[12,6]量化
        PHR_IQ = PHR_IQ.';
        [PHR_IQq] = quanti_txt(PHR_IQ,q_type_1,write);%量化并用于后续计算
        PHR_IQq = PHR_IQq.';
        %% 去除填充的比特
        com_out_PHR = Bit_Remove_Padding(PHR_IQq,PHR_CopyNum,PHR_BPC,PadBitsNum_PHR); 
        %[12,6]量化
        [com_out_PHRs] = quanti_txt(com_out_PHR,q_type_1,write); %量化并用于后续计算
        
        %% 信道解交织
        channal_de_inter_PHR = channal_deinter(PHR_pbsize,com_out_PHRs,PHR_rate); 
        q = quantizer('fixed','round','saturate',[8,6]);  %量化类型
        deinter_PHR_data = quantize(q,channal_de_inter_PHR); %输出数据进行8，6量化
        PHR_datain = deinter_PHR_data;
        
        %% 解扩运算第一步
        if isequal(SIG_data,[0,0,0]) || isequal(SIG_data,[0,0,1])
        % 若是PRDU帧则需要先解扩再译码再合并
            [despread_dout] = DSSS_Despread1(deinter_PHR_data,PRDU_pbsize*8,M); %解扩运算第一步
            PHR_datain = [despread_dout,deinter_PHR_data((PHR_pbsize*8+1):end)];
        end
        
        %% PHR Turbo译码
        tempPHR = PHR_datain > 0;  %01硬判决
        [decode_out_PHR,soft_out_PHR] = decoder_turbo(PHR_datain,iter_num,Lc,PHR_rate);  %decoder_algorithmturbo译码tempPHR
%         figure;stem(tempPHR(1:PHR_pbsize*8)-decode_out_PHR);
%         decode_out_PHR = tempPHR(1:PHR_pbsize*8);

%% --------------14.PRDU------------------
        % 根据SIG选择重传帧还是数据帧解码
        if isequal(SIG_data,[0,0,0]) || isequal(SIG_data,[0,0,1])
            decode_flag = 0;
        else
            decode_flag = 1;
        end
        
        if decode_flag == 0
            [PRDU_dout] = DSSS_Despread2(decode_out_PHR,PRDU_pbsize*8,M); %解扩运算第二步
            [errNum_PRDU,err_PRDU_n] = biterr(PRDU_dout,PRDU_orgi);
            err_PRDU(n) = err_PRDU(n) + err_PRDU_n; %PHR译码解扰去crc后误码率（长K-24）  
%             if(err_PRDU > 0)
%                 break;
%             end
        else

        PHR_descramb_out = scramb_code(decode_out_PHR);   %解扰
        [crc_decode_PHR, error_PHR] = crc_decode(PHR_descramb_out);  %crc校验
        
%--测试-- 在不知道发的是什么的情况下，可以直接判断PHR有无误码
%         PHR_Byte = bit_converge(PHR_descramb_out,PHR_pbsize); %将bit转化为字节
        decode_phr = crc_24(PHR_descramb_out(1:(PHR_pbsize*8-24)),24);
%         figure;stem(decode_phr-PHR_descramb_out);title("PHR CRC校验"); 
        
        tempPHRS = scramb_code(tempPHR(1:(PHR_pbsize*8)));
        decode_phr1 = crc_24(tempPHRS(1:(PHR_pbsize*8-24)),24);
%         figure;stem(decode_phr1-tempPHRS(1:PHR_pbsize*8));title("PHR No Turbo CRC校验");

%% --------------15.计算PHR err------------------
        [errNum_channel_PHR,err_channel_PHR_n] = biterr(channal_de_inter_PHR>0,PHR_turbo_out);
        err_channel_PHR(n) = err_channel_PHR(n) + err_channel_PHR_n; %PHR解交织后误码率（长N）
        [errNum_PHR,err_PHR_n] = biterr(crc_decode_PHR,PHR_data_orgi);
        err_PHR(n) = err_PHR(n) + err_PHR_n; %PHR译码解扰去crc后误码率（长K-24）  


%低信噪比加大阈值测试 
        
%         [errNum_PHR,err_PHR_n] = biterr(crc_decode_PHR,PHR_data_orgi);
%         if k<=total_number/2
%            err_PHR(n) = err_PHR(n) + err_PHR_n; %PHR译码解扰去crc后误码率（长K-24） 
%         else 
%            err_PHR(n+N) = err_PHR(n+N) + err_PHR_n; %PHR译码解扰去crc后误码率（长K-24）
%         end
%         if(err_PHR > 0)
%             break;
%         end
        end
        
         zhen = zhen + 1
    end
%% -------------打印本次snr下的err---------------------------
%     fp = fopen(filelocation,'a');
%     fprintf(fp,'%e   ',err_PHR(n)/total_number);
%     fclose(fp);
 
end
MSE_CFO = MSE_CFO ./total_number
err_channel_PHR = err_channel_PHR ./ total_number
err_PHR = err_PHR ./ total_number 
err_PRDU = err_PRDU ./ total_number 
err_SIG = err_SIG ./ total_number 
pick = pick ./ total_number

%低信噪比加大阈值测试 
% err_PHR = err_PHR ./ total_number/2 
% err_location = wr_location 
% 
figure;plot(pick,snr,'-ro');grid on;
xlabel('信噪比snr/dB');ylabel('估计的SNR');

figure;semilogy(snr,(err_PHR),'-bo');grid on;
xlabel('信噪比snr/dB');ylabel('PHR误码率');
% 
% figure;semilogy(snr,(err_PRDU),'-bo');grid on;
% xlabel('信噪比snr/dB');ylabel('PRDU误码率');

% figure;semilogy(snr,(err_PHR(1:N)),'-bo');
% hold on;semilogy(snr,(err_PHR(N+1:2*N)),'-ro');
% legendText = legend('阈值不变','阈值改变','FontSize', 10,'FontName','宋体');
% grid on;
% xlabel('信噪比snr/dB');ylabel('PHR误码率');
% 
% 
% figure;semilogy(snr,(err_location(1:N)),'-bo');
% hold on;semilogy(snr,(err_location(N+1:2*N)),'-ro');
% legendText = legend('阈值不变','阈值改变','FontSize', 10,'FontName','宋体');
% grid on;
% xlabel('信噪比snr/dB');ylabel('误同步次数');

%-----------------------------画图--------------------------------------
% figure;
% semilogy(snr, MSE_CFO,'-+');grid on;
% xlabel('snr/dB'), ylabel('delta f'); title('CFO 估计差值');
%% -------------打印--------------------------
% fp = fopen(filelocation,'a');
% fprintf(fp,'];\nerr_PHR = [');
% for i_snr=1:N
%     fprintf(fp,'%e   ',err_PHR(i_snr));
% end
% fprintf(fp,'];\nerr_SIG = [');
% for i_snr=1:N
%     fprintf(fp,'%e   ',err_SIG(i_snr));
% end
% fprintf(fp,'];\nCFO = [');
% for i_snr=1:N
%     fprintf(fp,'%e   ',MSE_CFO(i_snr));
% end
% fprintf(fp,'];\nerr_channelPSDUK = [');
% for i_snr=1:N
%     fprintf(fp,'%e   ',err_channel_PHR(i_snr));
% end
% fprintf(fp,'];\n------历时 %s秒',toc);
% fclose(fp);

% figure(1)
% hold on
% title("PHR数据误码图");
% stem(phr_dout1-decode_out_PHR);
% hold off