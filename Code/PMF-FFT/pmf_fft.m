function [peak_index,fre] = pmf_fft(data_syn_L,loc_L4)
%%pmf_fft粗捕获 适用于大频偏情况 对于信号做快速捕获 返回相应码相位和多普勒频偏范围

%%初始参数 64点fft  每段16个数据 分32段
N=128;
P=32;
n = 1;%初始化（同步起始位置）


N_data=length(data_syn_L);    %%数据长度
N_loc_S4=length(loc_L4);  %%前导长度为512
I = zeros(1, N);%I路测度函数
Q = zeros(1, N);%Q路测度函数
M = zeros(1, N);%组合测度函数

L=floor(N_data/(16*3));  %%过采样16倍 每次计算完一次数据移动16位  除以3表示先不计算所有数据减小计算量
% overlap = zeros(1, N); %%全部重叠的测度函数
norm = zeros(L,N); %%完全不重叠的测度函数

while  n < L
         data_temp = data_syn_L(1,(n-1)*16+1:(n-1)*16+N_loc_S4);  %%取出缓存在fifo中的数据
         for m=1:P                                       %%分段求相关
            I(m)=sum(real(data_temp((m-1)*16+1:m*16)).*loc_L4((m-1)*16+1:m*16));   %%I路数据
            Q(m)=sum(imag(data_temp((m-1)*16+1:m*16)).*loc_L4((m-1)*16+1:m*16));   %%Q路数据
         end
         M = I+Q;
%          abs(fft(M)).^2
         norm(n,:) = norm(n,:) + abs(fft(M)).^2;  
         n=n+1;
end

%%将频谱搬移到正常状态
for n = 1:L
    norm(n,:) =  fftshift(norm(n,:));
end

%频率分辨率计算：
%分辨率
f = 4160000/(16*N);

x = [1:	L];
y = [-N/2:N/2-1]*f; 
% figure(3)
% surf(y,x,abs(norm))
% ylabel('码相位')
% xlabel('多普勒频率（Hz）')
% title('捕获结果')

%%输出粗捕获结果  码相位目前一个单位代表16位长度 由过采样率决定
[a,b]=max(abs(norm(:)));
[m,n]=ind2sub(size(norm),b);

peak_index = (m-1)*16;
fre= -(n-64) * f;   %%移动频率至中心频点  目前分辨率位2000hz（f） 可通过调整参数改变

end