function [data_out] = combine_frame(data_in,CarrierNUM,N_FFT)

    % 前导长训练符号产生
    [LTF] = ltf_gen(CarrierNUM,N_FFT);  %根据可用子载波数生成前导
    
    % 组帧
    data_out = [LTF data_in];

end
