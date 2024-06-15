function [H_LS_EST,H_LS_EST_CAR] = channel_estimate(rx_pilot,loc_pilot,symbol_num,Np,N_FFT,Pilot_method,curr_pilot,ValidCarrierNum)
    
    int_opt = 'linear';

    H_LS = rx_pilot ./ loc_pilot;

    if lower(int_opt(1)) == 'l'
        method = 'linear'; 
    else
        method = 'spline'; 
    end

    if Pilot_method == 0

        %块状导频 暂时未做，之后补上

    else

        curr_data = curr_pilot(1):curr_pilot(end);  %给出数据所在的位置 虚拟子载波的位置填充0即可

        %去除导频所在的位置 获得数据子载波所在的位置
        for i = 1:Np
            curr_data(curr_data == curr_pilot(i)) = [];
        end
        curr_data(curr_data == N_FFT/2 + 1) = [];       %去除直流子载波所在位置 该位置填充0即可
    
        curr_ofdm = [curr_pilot(1):N_FFT/2,N_FFT/2+2:curr_pilot(end)];
    end

    H_LS_EST_CAR = zeros(symbol_num,length(curr_ofdm)); %数据载波+导频载波上的信道估计值
    H_LS_EST_ALL = zeros(symbol_num,N_FFT); %全部载波上的信道估计值 包括外插值
    H_LS_EST = zeros(symbol_num,length(curr_data)); %数据载波上的信道估计值

    % 线性内插
    for i = 1 : symbol_num
        H_LS_EST_CAR(i,:) = interp1(curr_pilot,H_LS(i,:),curr_ofdm,method);
    end

    curr_ofdm_all = 1:32;
    % 获得数据载波上的信道估计值
    for j = 1 : symbol_num
        H_LS_EST_ALL(j,:) = interp1(curr_pilot,H_LS(j,:),curr_ofdm_all,method);
    end

    for m = 1 : ValidCarrierNum
        H_LS_EST(:,m) = H_LS_EST_ALL(:,curr_data(m));
    end

end
