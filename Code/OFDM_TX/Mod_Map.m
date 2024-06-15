function [MOD_OUT,wlanSymMap] = Mod_Map(BPC,data_in,ValidCarrierNum,symbol_num)

    %调制阶数
    M = 2 ^ BPC;

    %星座映射
    % symOrder = 'gray';                          % 使用格雷码
    % dataMod = qammod(data_in, M, symOrder, 'UnitAveragePower', true);     %Se modula a M-QAM

    wlanSymMap = randperm(M,M) - 1;   %生成M以内的n个不重复的随机数并减1转化为0-(M-1)的整数

    if BPC == 1

        K_mod = 1;
        dataMod = qammod(data_in,M,'UnitAveragePower', true,'InputType', 'bit');

    else

        K_mod = sqrt(3/2/(M-1));                % M-QAM 归一化因子
        % dataMod = qammod(data_in,M,wlanSymMap,'UnitAveragePower', true,'PlotConstellation',true);
        dataMod = qammod(data_in,M,wlanSymMap,'UnitAveragePower', true,'InputType', 'bit');

    end

    
    %绘制星座图
    % scatterplot(dataMod); axis([-2 2 -2 2]);

    MOD_OUT = reshape(dataMod,ValidCarrierNum,symbol_num).';

end
