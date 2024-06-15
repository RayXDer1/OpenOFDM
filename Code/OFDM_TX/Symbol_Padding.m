function [data_out,symbol_num,PadBits_Num] = Symbol_Padding(data_in,ValidCarrierNum,BPC)
%数据填充模块

    %每个ofdm符号的比特数
    Bits_PerOFDM = BPC * ValidCarrierNum;
    Bits_PerGroup = Bits_PerOFDM;
    
    %计算数据长度
    data_len = length(data_in);
    
    %最后一个OFDM符号内的比特数
    Bits_LastOFDM = data_len - Bits_PerOFDM * fix(data_len / Bits_PerOFDM);
    
    %计算最后一个组中的bit数
    if Bits_LastOFDM == 0
        Bits_LastGroup = Bits_PerOFDM;
    else
        Bits_LastGroup = Bits_LastOFDM - Bits_PerOFDM * fix((Bits_LastOFDM - 1) / Bits_PerOFDM);
    end
    
    % 计算填充比特数
    PadBits_Num = Bits_PerGroup - Bits_LastGroup;
    
    %计算符号数
    symbol_num = ceil(data_len / Bits_PerOFDM);
    
    %填充PadBits_Num个数据
    pad_data = [data_in round(rand(1,PadBits_Num))];
    
    %按照BPC对数据进行分组
    reshape_data = reshape(pad_data,BPC,length(pad_data)/BPC).';
    
    data_out = reshape_data.';

end
