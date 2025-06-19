%导频是一个PN10序列
%生成多项式：x^10 + x^3 + 1
function [Pilot_out,scramble_patt] = ADD_Pilot(symbol_num,IQ_OUT)
r=10;  %指定寄存器位数，也就确定了m序列的长度
g=1*ones(1,r)  ;
PN_out = zeros(1,2^r-1);
for k=1 :(2^r-1)
    PN_out(k)=g(r);%%%out
    tmp = xor(g(3),g(r));%将3 10 两位进行异或运算
    g(2:r)=g(1:r-1);
    g(1)=tmp;%将寄存器的最后一位放到第一位，进行下一次移位
end
% 扰码使用BPSK映射
PN_BPSK = BPSK(PN_out);
%test 
PN_BPSK = [PN_BPSK(11:1023),PN_BPSK(1:10)];
%% 导频载波数&所有载波数
carr_num = 32;
pilot_set_num = 4;
Pilot_carr_num = 5;
scramble_patt = repmat(PN_BPSK,1,ceil(symbol_num * Pilot_carr_num/length(PN_BPSK)));%重复导频扰码，使得其长度至少与OFDM符号数一样

%% 插入导频
IQ_OUT = IQ_OUT.';  %给映射后数据转置
mod_ofdm_syms = zeros(carr_num,size(IQ_OUT,2));%全部子载波数*导频组数

%---------------导频组-------------------

    scramble_patt = scramble_patt(1:size(IQ_OUT,2)*5); % option每个符号有2个子载波插入导频
    scramble_patt = 10^(2/20)*real(scramble_patt);  %功率控制sqrt(carr_num)*
    scramble_patt = reshape(scramble_patt,5,size(IQ_OUT,2));
    left_symbols = size(IQ_OUT,2) - pilot_set_num*fix(size(IQ_OUT,2)/pilot_set_num);
    for column = 0:pilot_set_num:pilot_set_num*fix(size(IQ_OUT,2)/pilot_set_num)-pilot_set_num
        % 插入导频
        mod_ofdm_syms([7 11 15 20 24],1+column) = scramble_patt(:,1+column);
        mod_ofdm_syms([8 12 16 21 25],2+column) = scramble_patt(:,2+column);
        mod_ofdm_syms([9 13 18 22 26],3+column) = scramble_patt(:,3+column);
        mod_ofdm_syms([10 14 19 23 27],4+column) = scramble_patt(:,4+column);
        % 插入数据
        mod_ofdm_syms([8:10 12:14 16 18:19 21:23 25:27],1+column) = IQ_OUT(:,1+column);
        mod_ofdm_syms([7 9:10 11 13:14 15 18:19 20 22:23 24 26:27],2+column) = IQ_OUT(:,2+column);
        mod_ofdm_syms([7:8 10 11:12 14 15:16 19 20:21 23 24:25 27],3+column) = IQ_OUT(:,3+column);
        mod_ofdm_syms([7:9 11:13 15:16 18 20:22 24:26],4+column) = IQ_OUT(:,4+column);
    end
    if left_symbols == 1
        mod_ofdm_syms([7 11 15 20 24],1+size(IQ_OUT,2)-left_symbols) = scramble_patt(:,1+size(IQ_OUT,2)-left_symbols);
        % 插入数据
        mod_ofdm_syms([8:10 12:14 16 18:19 21:23 25:27],1+size(IQ_OUT,2)-left_symbols) = IQ_OUT(:,1+size(IQ_OUT,2)-left_symbols);
    elseif left_symbols == 2
        % 插入导频
        mod_ofdm_syms([7 19],1+size(IQ_OUT,2)-left_symbols) = scramble_patt(:,1+size(IQ_OUT,2)-left_symbols);
        mod_ofdm_syms([15 27],2+size(IQ_OUT,2)-left_symbols) = scramble_patt(:,2+size(IQ_OUT,2)-left_symbols);
        % 插入数据
        mod_ofdm_syms([8:16 18 20:27],1+size(IQ_OUT,2)-left_symbols) = IQ_OUT(:,1+size(IQ_OUT,2)-left_symbols);
        mod_ofdm_syms([7:14 16 18:26],2+size(IQ_OUT,2)-left_symbols) = IQ_OUT(:,2+size(IQ_OUT,2)-left_symbols);
    end    

%% 导频插入后的数据输出
%  Pilot_out = mod_ofdm_syms(:);
Pilot_out = mod_ofdm_syms.';
end
