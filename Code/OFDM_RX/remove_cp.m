function data_out = remove_cp(data_in,symbol_num,N_FFT,N_CP)

    data_out = zeros(symbol_num,N_FFT);

    CP_DATA = reshape(data_in,N_FFT+N_CP,symbol_num);
    CP_DATA_TRS = CP_DATA.';

    % 添加循环前缀
    for i = 1:symbol_num
        data_out(i,:) = CP_DATA_TRS(i,N_CP+1:end);
    end    

end
