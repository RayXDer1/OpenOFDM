function [HPREAMBLE_LS,HPREAMBLE_PLS,rx_ofdm_data,rx_pofdm_data,synchron_local,synchron_proba,MSE_cfo] = rx_lscnn_test1(synchron_local,synchron_proba,MSE_cfo,rx_signal,rx_perfect_signal,CarrierNUM,symbol_num,N_FFT,N_CP,Nsym_CP,cfo_offset,Fs,n)

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
    
    loc_index = [2:16 18:32];
    % loc_index = 1:N_FFT;
    rx_fft_pilot = fft(rx_preamble_data(end-31:end))./sqrt(32); 
    loc_fft_pilot = fft(LOC_L)./sqrt(32); 
    HPREAMBLE_LS = rx_fft_pilot(loc_index) ./ loc_fft_pilot(loc_index);


    %对无噪声的数据进行同样的流程
    % cfo_estimate_value = cfo_offset;
    rx_psynchron_signal = rx_perfect_signal(1,peak_index_L:end); %同步到的完美OFDM信号
    % 根据计算出的频偏值进行补偿和校正
    timebase = (0:length(rx_psynchron_signal)-1);
    rx_pcompen_signal = exp(1j*(2*pi*(-cfo_estimate_value)*timebase/Fs)).*rx_psynchron_signal;
    %输出数据
    rx_pofdm_cpdata = rx_pcompen_signal(1,N_LTF+1:N_LTF+N_PSDU);
    rx_preamble_pdata = rx_pcompen_signal(1,1:4*N_FFT);
    % ----------------------------去循环前缀-----------------------------
    [rx_pofdm_data] = remove_cp(rx_pofdm_cpdata,symbol_num,N_FFT,N_CP);

    rx_pfft_pilot = fft(rx_preamble_pdata(end-31:end))./sqrt(32); 
    loc_pfft_pilot = fft(LOC_L)./sqrt(32); 
    HPREAMBLE_PLS = rx_pfft_pilot(loc_index) ./ loc_pfft_pilot(loc_index);

end
