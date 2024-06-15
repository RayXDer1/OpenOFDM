function [Pilot_Insert_out,loc_pilot,phase] = Pilot_Insert(MOD_OUT,symbol_num,Pilot_method,VCNum_method,Np,N_FFT,ValidCarrierNum)

    num = [-3,-3,4,3,3,0,-3,-2,-2,-3,3,-2,1,1,-2,3,...
                 1,-2,0,2,-1,1,4,1,2,4,4,2,4,-2,1,2];
    phase = num .* pi/4;

    pilot_value = [1 -1 1 1 -1 1];
    %参数初始化
    pilot_pos = [];
    ofdm_symbol = zeros(symbol_num,N_FFT);  %全部子载波数*符号数
    loc_pilot = zeros(symbol_num,Np);  %全部子载波数*符号数
    
    pilot_pos(1,1:2) = [-10 10] + N_FFT/2 + 1;
    pilot_pos(2,1:4) = [-11 -4 4 11] + N_FFT/2 + 1;
    pilot_pos(3,1:6) = [-12 -8 -4 4 8 12] + N_FFT/2 + 1;
    pilot_pos(4,1:2) = [-13 13] + N_FFT/2 + 1;
    pilot_pos(5,1:4) = [-14 -5 5 14] + N_FFT/2 + 1;
    pilot_pos(6,1:6) = [-15 -9 -3 3 9 15] + N_FFT/2 + 1;
    
    if Pilot_method == 0
        %块状导频 暂时未做，之后补上
    else
        curr_pilot = pilot_pos(VCNum_method * 3 + Pilot_method,1:Np);
        curr_data = curr_pilot(1):curr_pilot(end);  %给出数据所在的位置 虚拟子载波的位置填充0即可
    
        %去除导频所在的位置 获得数据子载波所在的位置
        for i = 1:Np
            curr_data(curr_data == curr_pilot(i)) = [];
        end
        curr_data(curr_data == N_FFT/2 + 1) = [];       %去除直流子载波所在位置 该位置填充0即可
    
        for j = 1:Np
            ofdm_symbol(:,curr_pilot(j)) = pilot_value(j);
        end
    
        for m = 1:ValidCarrierNum
            ofdm_symbol(:,curr_data(m)) = MOD_OUT(:,m);
        end
    
    end

    for row = 1 : symbol_num
        for col = 1 : N_FFT
            ofdm_symbol(row,col) = ofdm_symbol(row,col) .* exp(1j * phase(col));      
        end
    end
    ofdm_symbol_trs = ofdm_symbol.';
    
    %子载波调换
    %对后的数据进行调换，对齐载波号，与IFFT的频点相互对应：
    %0,1,2,...,8,...,15,-16,-15,..,-8,...,-3,-2,-1
    ofdm_symbol_change(1:N_FFT/2,:) = ofdm_symbol_trs((N_FFT/2+1):N_FFT,:);
    ofdm_symbol_change((N_FFT/2+1):N_FFT,:) = ofdm_symbol_trs(1:N_FFT/2,:);
    
    Pilot_Insert_out = ofdm_symbol_change.';
    
    for p = 1 : symbol_num
        loc_pilot(p,:) = pilot_value(1:Np);
    end

end
