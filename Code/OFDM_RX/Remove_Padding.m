function [data_out] = Remove_Padding(data_in,PadBits_Num)

    % 将解调输出的2进制转化为行向量
    bit_data_out = data_in(:);
    bit_data_out_trs = bit_data_out.';

    data_out = bit_data_out_trs(1:(end - PadBits_Num));

end
