function [rx_data,rx_pilot,curr_pilot] = pilot_extract(data_in,symbol_num,Pilot_method,VCNum_method,Np,N_FFT,ValidCarrierNum,phase)

    %参数初始化
    pilot_pos = [];

    rx_data = zeros(symbol_num,ValidCarrierNum);  %数据子载波数*符号数
    rx_pilot = zeros(symbol_num,Np);  %数据子载波数*符号数
    
    pilot_pos(1,1:2) = [-10 10] + N_FFT/2 + 1;
    pilot_pos(2,1:4) = [-11 -4 4 11] + N_FFT/2 + 1;
    pilot_pos(3,1:6) = [-12 -8 -4 4 8 12] + N_FFT/2 + 1;
    pilot_pos(4,1:2) = [-13 13] + N_FFT/2 + 1;
    pilot_pos(5,1:4) = [-14 -5 5 14] + N_FFT/2 + 1;
    pilot_pos(6,1:6) = [-15 -9 -3 3 9 15] + N_FFT/2 + 1;
    data_in_trs = data_in.';

    %对数据进行调换，对齐载波号，与数据频点相互对应：
    %-16,-15,..,-8,...,-3,-2,-1,0,1,2,...,8,...,15 

    ofdm_symbol_change(1:N_FFT/2,:)  = data_in_trs((N_FFT/2+1):N_FFT,:);
    ofdm_symbol_change((N_FFT/2+1):N_FFT,:) = data_in_trs(1:N_FFT/2,:);

    ofdm_symbol_change_trs = ofdm_symbol_change.';

    for row = 1 : symbol_num
        for col = 1 : N_FFT
            ofdm_symbol_change_trs(row,col) = ofdm_symbol_change_trs(row,col) .* exp(-1j * phase(col));      
        end
    end

    if Pilot_method == 0
        %块状导频 暂时未做，之后补上
    else
        curr_pilot = pilot_pos(VCNum_method * 3 + Pilot_method,1:Np);
        curr_data = curr_pilot(1):curr_pilot(end);  %给出数据所在的位置 虚拟子载波为0

        %去除导频所在的位置 获得数据子载波所在的位置
        for i = 1:Np
            curr_data(curr_data == curr_pilot(i)) = [];
        end
        curr_data(curr_data == N_FFT/2 + 1) = [];       %去除直流子载波所在位置 该位置填充0即可

        for j = 1:Np
            rx_pilot(:,j) = ofdm_symbol_change_trs(:,curr_pilot(j));
        end

        for m = 1:ValidCarrierNum
            rx_data(:,m) = ofdm_symbol_change_trs(:,curr_data(m));
        end

    end

end