
function [D1,symbol_num,ValidCarrierNum,PadBitsNum] = Bit_Padding(Data,CopyNum,BPC)
%Data为输入串行数流
% CopuNum为拷贝次数
% BPC为每个载波调制数据个数
% InterNum为交织器个数
% m输出为载波形式

% ==================================各项参数计算=================================
ValidCarrierNum = 15;
%% copynum为1的情况
if CopyNum == 1
    carr_num = ValidCarrierNum;
    UsedCarrierNum = ValidCarrierNum;
    [CopyData,PadBitsNum] = CopyAndPad(Data,CopyNum,UsedCarrierNum,BPC);
    if (BPC == 1)
      [~,col] = size(CopyData);
      data_iq = reshape(CopyData,1,col);
    elseif (BPC == 2)
      [~,col] = size(CopyData);
      data_iq = reshape(CopyData,2,col/2);
    else
      [~,col] = size(CopyData);
      data_iq = reshape(CopyData,4,col/4);
    end
    [~,b] = size(data_iq);
    copy_length =  b ;
    data_copy1 = data_iq(:,1:copy_length);
    out_data1 =  data_copy1;
    pyload_diver_out = out_data1;
    symbol_num = length(pyload_diver_out)/UsedCarrierNum;
    
end
%BPC=1,调制方式为BPSK，数据为一路
%BPC=2,调制方式为QPSK，先把数据分成IQ两路
%BPC=4，调制方式为16QAM,每个载波对应4比特数据
%   [~,col] = size(CopyData);
%   data_iq = reshape(CopyData,2,col/BPC);
%% 
D1 = zeros(symbol_num*BPC,ValidCarrierNum);  % 当UsedCarrierNum不等于ValidCarriernNum时，使用底位的子载波
if BPC==4
  for i = 1 : 4 : symbol_num*BPC
      D1(i:i+3,1:UsedCarrierNum) = pyload_diver_out(:,(((i+3)/4)-1)*carr_num+1:(i+3)/4*carr_num);
  end
  
elseif BPC==2
  for i = 1 : 2 : symbol_num*BPC
      D1(i:i+1,1:UsedCarrierNum) = pyload_diver_out(:,(((i+1)/2)-1)*carr_num+1:(i+1)/2*carr_num);
  end
  
else
  for i = 1 : 1 : symbol_num*BPC
      D1(i,1:UsedCarrierNum) = pyload_diver_out(:,(i-1)*carr_num+1:i*carr_num);
  end
end    
end