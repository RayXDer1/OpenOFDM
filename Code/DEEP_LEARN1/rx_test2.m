function [PSDU_decode_out] = rx_test2(rx_ofdm_data,symbol_num,Pilot_method,VCNum_method,Np,loc_pilot,phase,N_FFT,ValidCarrierNum,PSDU_BPC,wlanSymMap,PadBits_Num,PSDU_PB)   

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

end