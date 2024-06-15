%加噪声
function [noise_signal_out,perfect_signal_out,h_out] = add_channel(signal_in,snr,cfo_offset,Fs,frame_cnt)

    
    %----------施加频偏---------------------
    N_signal_in = length(signal_in);
    time_base = 0 : (N_signal_in-1);
    rad_offset = 2*pi*(cfo_offset/(Fs).*(time_base));
    exp_Freq = exp(1i*rad_offset);
    RX_Signal1 = signal_in.*exp_Freq; 

    %----------施加AWGN信道---------------------
    Multitap_h1 = 1;

    %----------施加单径信道---------------------
    Multitap_h2 = (randn + 1j * randn);
    Multitap_h2 = Multitap_h2/norm(Multitap_h2) / (1.9375 * sqrt(pi/2));

    %----------施加两径信道---------------------
    Multitap_h3 = [(randn + 1j * randn);...
                   (randn + 1j * randn) / 8] / (1.9375 * sqrt(pi/2));
    Multitap_h3 = Multitap_h3/norm(Multitap_h3);
    Multitap_h3 = Multitap_h3.';

    xindex = mod(frame_cnt,9);

    if (xindex <= 2)
        Multitap_ho = Multitap_h1;
    elseif (xindex > 2) && (xindex <= 5)
        Multitap_ho = Multitap_h2;
    else
        Multitap_ho = Multitap_h3;
    end

    comp_len = 200;
    P1 = zeros(1,comp_len);       %在帧前加空隙200
    P2 = zeros(1,comp_len);       %在帧后加空隙200
    RX_Signal2 = [P1,RX_Signal1,P2];
    RX_Signal3 = conv(RX_Signal2,Multitap_ho);
    
    %无信道衰落
%     RX_Signal3 = RX_Signal1;
    
    %----------计算信号、噪声功率------------
    sigPower = var(RX_Signal3);         % 单径发送信号功率
    SNR = 10 ^ (snr/10);                % 将信噪比转换位标准单位    
    Noise_Power = sigPower / SNR;       % 计算噪声功率
    sigma = sqrt(Noise_Power/2);        % 计算均方值
    
    %----------施加随机噪声信号-------------
    Channel_noise = sigma * randn(1, length(RX_Signal3)) + 1i * sigma * randn(1, length(RX_Signal3));      %产生噪声
    RX_Signal4 = RX_Signal3 + Channel_noise;               %加噪声

    noise_signal_out = RX_Signal4;
    perfect_signal_out = RX_Signal3;

    %将在时域上建模的信道h导出在频域上
    h_out = fft(Multitap_ho,32);
    h_out = [h_out(17:32) h_out(1:16)]; %调换子载波
   
end