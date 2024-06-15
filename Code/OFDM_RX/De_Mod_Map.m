function [DEMOD_OUT] = De_Mod_Map(BPC,data_in,wlanSymMap)

    data_in_trs = data_in.';
    demod_data = data_in_trs(:);
    demod_data_trs = demod_data.';

    M = 2 ^ BPC;
    % 解调输出10进制数
    % DEMOD_OUT = qamdemod(demod_data_trs,M,wlanSymMap,'UnitAveragePower',true);

    % 解调时直接转化为2进制，可以省去之后的10进制转换2进制数组的过程
    if BPC == 1
        DEMOD_OUT = qamdemod(demod_data_trs,M,'UnitAveragePower',true,'OutputType', 'bit'); 
    else
        DEMOD_OUT = qamdemod(demod_data_trs,M,wlanSymMap,'UnitAveragePower',true,'OutputType', 'bit'); 
    end

end
