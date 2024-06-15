function [data_out] = add_cp(data_in,symbol_num,N_FFT,N_CP)

    CP_DATA_OUT = zeros(symbol_num,N_FFT+N_CP);

    % 添加循环前缀
    for i = 1:symbol_num
        CP_DATA = data_in(i,N_FFT - N_CP + 1 : N_FFT);
        CP_DATA_OUT(i,:) = [CP_DATA, data_in(i,:)];
    end

    CP_ADD_DATA = CP_DATA_OUT.';
    data_out = CP_ADD_DATA(:).';

end
