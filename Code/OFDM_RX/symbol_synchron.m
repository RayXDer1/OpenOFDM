function peak_index = symbol_synchron(data_in,loc_L,w)
% 利用LTF进行互相关符号同步算法
% 输入待检测OFDM信号data_in，检测依据：本地前导信号loc_L
% 输出峰值坐标peak_index

LEN = 1;
noise_threshold = 100;              %噪声门限值: 100
corr_cnt = 1;                       %捕获次数
n = 1;                              %初始化（同步起始位置）
LEN_loc_L = length(loc_L);
detct_inteval = LEN_loc_L - 2 * LEN;%最大值检测区间
N = length(data_in);
mean_value = single(zeros(1, N));
MAX_INDEX = 400;                    %最大检测样点
peak_index = ones(1, 4) * MAX_INDEX;

% 计算互相关峰
[x_corr_data,~] = xcorr(single(data_in),single(loc_L));
peak_M = abs(x_corr_data) .^ 2;
M = peak_M(N:end);                  %相关峰测度函数

while  n < N - LEN_loc_L + 1

        % 对峰值取LEN_loc_L点滑窗和 并与w相乘 作为阈值检测条件
        if(n > LEN_loc_L)
            mean_value(n-detct_inteval/2) = w * sum(M(n-LEN_loc_L:n-1))./LEN_loc_L;
        end

        % 若为峰值误检，则通过以下策略纠错
        if n > detct_inteval

            if(M(n-detct_inteval/2) == max(M(n-detct_inteval+1:n))) && M(n-detct_inteval/2) > mean_value(n-detct_inteval/2) && M(n-detct_inteval/2) >= noise_threshold
                
                % 检测到峰值则记录
                peak_index(corr_cnt) = n - detct_inteval/2;
                corr_cnt = corr_cnt + 1;
               
                % 当峰值出现误检和漏检时采取纠错策略
                if(((corr_cnt == 3) && (abs(peak_index(2) - peak_index(1) - LEN_loc_L) > LEN)))
                    corr_cnt = 2;
                    peak_index(1) =  peak_index(2);
                elseif(((corr_cnt == 4) && (abs(peak_index(3) - peak_index(2) - LEN_loc_L) > LEN)))
                    corr_cnt = 2;
                    peak_index(1) =  peak_index(3);
                end

            end

            %满足检测条件则退出峰值检测
            if (((corr_cnt >= 4) && (n > MAX_INDEX)) || (n > MAX_INDEX))
                break;
            end

        end
         
        n = n + 1;
end

% 画图%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;plot(M,'LineWidth',1.5);
% hold on;plot(mean_value,'LineWidth',1.5);
% hold on;plot(peak_index,M(peak_index),'ro');
% title("符号同步相关峰图");
% legend('相关峰','阈值','峰值点')
% xlabel('样点数');ylabel('相关峰测度函数');
% hold off;

end