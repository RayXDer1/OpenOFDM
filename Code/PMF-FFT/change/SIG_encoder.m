%SIG编码
%(36,3)分组码
function [SIG_out,SIG_carr_num,SIG_symbol_num] = SIG_encoder(m)
% ------------------------生成矩阵---------------------
G=[1 0 1 0 0 0 1 1 0 0 1 1 1 0 0 1 1 0 1 1 0 0 1 0 0 1 1 1 1 1 0 1 1 0 1 1;...
    0 1 1 0 1 0 0 1 1 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 0 0 0 0 1 0 1 0 0 0 1 1;...
    0 0 1 1 1 1 0 0 1 0 0 0 1 0 1 1 1 0 1 0 1 0 1 1 1 1 0 1 0 0 1 0 1 1 1 0];
[k,n]=size(G);%生成矩阵的行数是信息位数 列数是编码位数
r=n-k;

l=length(m);
if(mod(l,k))
    disp('输入的信息有误')%一组信息长度为k 总信息位l 必须为k的整数倍
else
    num=l/k;%num表示总共传递的信息组数
end
%将输入信号的行向量转化为矩阵，矩阵每一行为一组信息
m_col=[];
for i=1:num
    m_col(i,:)=m(k*(i-1)+1:i*k);
end
m=m_col;
%求监督矩阵H
encoded_sig=mod(m*G,2);%出现2要取余变0 编码矩阵A=m*G
Q=G(:,k+1:n);%生成矩阵G=[Ik|Q]
H=[Q',eye(r)];%H=[P|Tr]其中P=Q'
% disp('编码矩阵');encoded_sig
% disp('监督矩阵');H
%% ---------------------符号填充------------------------
SIG_pad = symbol_padding(encoded_sig);
%% --------------------加扰----------------------------
SIG_scramb_out = scramb_code(SIG_pad);
%% -------------------组帧---------------------
SIG_carr_num = 15;
SIG_symbol_num = 3;
SIG_total_num = SIG_carr_num*SIG_symbol_num;
SIG_out = repmat(SIG_scramb_out,1,ceil(SIG_total_num/length(SIG_pad)));
SIG_out = SIG_out(1,1:SIG_total_num);
SIG_out = reshape(SIG_out,SIG_carr_num,SIG_symbol_num);
SIG_out=SIG_out.';

end   