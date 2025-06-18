clc;
clear all;

num = 10000;       %数据数量  
N = 20;          %子载波数量
length_snr = 5;

H_s1_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s1_r.xls');
H_s1_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s1_i.xls');
H_p1_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p1_r.xls');
H_p1_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p1_i.xls');

H_s2_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s2_r.xls');
H_s2_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s2_i.xls');
H_p2_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p2_r.xls');
H_p2_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p2_i.xls');

H_s3_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s3_r.xls');
H_s3_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s3_i.xls');
H_p3_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p3_r.xls');
H_p3_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p3_i.xls');

H_s4_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s4_r.xls');
H_s4_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s4_i.xls');
H_p4_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p4_r.xls');
H_p4_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p4_i.xls');

H_s5_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s5_r.xls');
H_s5_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_s5_i.xls');
H_p5_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p5_r.xls');
H_p5_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\H_p5_i.xls');

H_s_r = zeros(5 * length_snr * num,N);
H_s_i = zeros(5 * length_snr * num,N);

H_p_r = zeros(5 * length_snr * num,N);
H_p_i = zeros(5 * length_snr * num,N);

for j = 1 : 1 : 50000
    H_s_r(j,:) = H_s1_r(j,:);
    H_s_r(50000 * 1 + j,:) = H_s2_r(j,:);
    H_s_r(50000 * 2 + j,:) = H_s3_r(j,:);
    H_s_r(50000 * 3 + j,:) = H_s4_r(j,:);
    H_s_r(50000 * 4 + j,:) = H_s5_r(j,:);

    H_s_i(j,:) = H_s1_i(j,:);
    H_s_i(50000 * 1 + j,:) = H_s2_i(j,:);
    H_s_i(50000 * 2 + j,:) = H_s3_i(j,:);
    H_s_i(50000 * 3 + j,:) = H_s4_i(j,:);
    H_s_i(50000 * 4 + j,:) = H_s5_i(j,:);

    H_p_r(j,:) = H_p1_r(j,:);
    H_p_r(50000 * 1 + j,:) = H_p2_r(j,:);
    H_p_r(50000 * 2 + j,:) = H_p3_r(j,:);
    H_p_r(50000 * 3 + j,:) = H_p4_r(j,:);
    H_p_r(50000 * 4 + j,:) = H_p5_r(j,:);

    H_p_i(j,:) = H_p1_i(j,:);
    H_p_i(50000 * 1 + j,:) = H_p2_i(j,:);
    H_p_i(50000 * 2 + j,:) = H_p3_i(j,:);
    H_p_i(50000 * 3 + j,:) = H_p4_i(j,:);
    H_p_i(50000 * 4 + j,:) = H_p5_i(j,:);
end

H_1_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_1_r.xls');
H_2_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_2_r.xls');
H_3_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_3_r.xls');
H_4_r = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_4_r.xls');

H_1_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_1_i.xls');
H_2_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_2_i.xls');
H_3_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_3_i.xls');
H_4_i = readmatrix('E:\Project\MATLAB\HuaWei_WuXian\WuXian_spread_V7\WuXian\FSR_NET\数据集\原始信道\H_4_i.xls');

H_r = zeros(4 * 5 * num,N);
H_i = zeros(4 * 5 * num,N);

for j = 1 : 1 : 50000
    H_r(j,:) = H_1_r(j,:);
    H_r(50000 * 1 + j,:) = H_2_r(j,:);
    H_r(50000 * 2 + j,:) = H_3_r(j,:);
    H_r(50000 * 3 + j,:) = H_4_r(j,:);

    H_i(j,:) = H_1_i(j,:);
    H_i(50000 * 1 + j,:) = H_2_i(j,:);
    H_i(50000 * 2 + j,:) = H_3_i(j,:);
    H_i(50000 * 3 + j,:) = H_4_i(j,:);
end


%数据拼接
trainRatio = 0.9;   %设置 90% 数据用于训练，剩余用于测试

H_data_1 = zeros(trainRatio * num * length_snr * 5,N * 2);
H_data_2 = zeros((num - trainRatio * num) * length_snr * 5,N * 2);
flag_data_1 = zeros(trainRatio * num * length_snr * 5,N * 2);
flag_data_2 = zeros((num - trainRatio * num) * length_snr * 5,N * 2);

for j = 1 : 1 : length_snr
    for i = 1 : 1 : trainRatio * num * 5
        for k = 1 : 1 : N
            I = i + (j - 1) * trainRatio * num * 5;
            J = i + (j - 1) * num * 5;
            H_data_1(I,2 * k - 1) = H_p_r(J,k);
            H_data_1(I,2 * k) = H_p_i(J,k);
        end
    end
end

for j = 1 : 1 : length_snr
    for i = (trainRatio * num * 5 + 1) : 1 : num * 5
        for k = 1 : 1 : N
            I = i - trainRatio * num * 5 + (j - 1) * (num * 5 - trainRatio * num * 5);
            J = i + (j - 1) * num * 5;
            H_data_2(I,2 * k - 1) = H_p_r(J,k);
            H_data_2(I,2 * k) = H_p_i(J,k);
        end
    end
end
    
% for i = 1 : 1 : num
%     % data_in(i,:) = [X_data(i,:),Y_data(i,:),H_data(i,:)];  %三路拼接输入
%     data_in(i,:) = H_data(i,:);                             %仅用H_LS作为输入
% end
%信道数据拼接
H_fix_r = zeros(num * 5,4 * N);
H_fix_i = zeros(num * 5,4 * N);
    
for i = 1 : 1 : num * 5
    I = 4 * (i - 1);
    H_fix_r(i,:) = [H_r(I + 1,:) H_r(I + 2,:) H_r(I + 3,:) H_r(I + 4,:)];
    H_fix_i(i,:) = [H_i(I + 1,:) H_i(I + 2,:) H_i(I + 3,:) H_i(I + 4,:)];
end

H_fix = zeros(num * 5,N * 8);
for i = 1 : 1 : num * 5
    for j = 1 : 1 : N * 4
        H_fix(i,2 * j - 1) = H_fix_r(i,j);
        H_fix(i,2 * j) = H_fix_i(i,j);
    end
end

H_train = cat(1,H_fix(1 : 45000,:),H_fix(1 : 45000,:),H_fix(1 : 45000,:),H_fix(1 : 45000,:),H_fix(1 : 45000,:));
H_test = cat(1,H_fix(45001 : 50000,:),H_fix(45001 : 50000,:),H_fix(45001 : 50000,:),H_fix(45001 : 50000,:),H_fix(45001 : 50000,:));


%训练测试数据分割
% trainCount = floor(num * trainRatio);
% X_train = data_in(1 : trainCount,:);
% X_test = data_in(trainCount + 1:num,:);
X_train = H_data_1;
X_test = H_data_2;

X_train = X_train';
X_test = X_test';

M_size = size(X_train, 2);
N_size = size(X_test, 2);

Y_train = flag_data_1;
Y_test = flag_data_2;

Y_train = Y_train';
Y_test = Y_test';

H_train = H_train';
H_test = H_test';

%数据转换
XTrainMapD=reshape(X_train,[size(X_train,1) / 8,8,1,M_size]);%训练集输入
XTestMapD =reshape(X_test,[size(X_test,1) / 8,8,1,N_size]); %测试集输入

HTrainMapD=reshape(H_train,[size(H_train,1) / 16,16,1,M_size]);%训练集输入
HTestMapD =reshape(H_test,[size(H_test,1) / 16,16,1,N_size]); %测试集输入

%神经网络构建
% layers = [
%     imageInputLayer([size(X_train,1) / 8, 8, 1])    % 输入层，输入长度为2*N
% 
%     convolution2dLayer([3, 3], 6,'Padding',2)     % 卷积层，4为滤波器宽度，1为滤波器数量
%     batchNormalizationLayer   
%     reluLayer                            % relu激活函数
% 
%     convolution2dLayer([3, 3], 3,'Padding',2)     % 卷积层，16为滤波器宽度，5为滤波器数量
%     batchNormalizationLayer   
%     reluLayer                            % relu激活函数
% 
%     convolution2dLayer([3, 3], 3,'Padding',2)     % 卷积层，16为滤波器宽度，5为滤波器数量
%     batchNormalizationLayer   
%     reluLayer
% 
%     convolution2dLayer([3, 3], 3,'Padding',1)     % 卷积层，16为滤波器宽度，5为滤波器数量
%     batchNormalizationLayer   
%     reluLayer                            % relu激活函数
% 
%     convolution2dLayer([3, 3], 3,'Padding',1)     % 卷积层，16为滤波器宽度，5为滤波器数量
%     batchNormalizationLayer   
%     reluLayer                            % relu激活函数
% 
%     convolution2dLayer([3, 3], 3,'Padding',1)     % 卷积层，16为滤波器宽度，5为滤波器数量
%     batchNormalizationLayer   
%     reluLayer                            % relu激活函数
% 
%     %信道矩阵扩充，由5 * 8到10 * 16
%     convolution2dLayer([3, 3], 6,'Padding',[1 0 2 2])     
%     batchNormalizationLayer   
%     reluLayer    
% 
%     convolution2dLayer([3, 3], 6,'Padding',2)     
%     batchNormalizationLayer   
%     reluLayer
% 
%     convolution2dLayer([3, 3], 6,'Padding',2)     
%     batchNormalizationLayer   
%     reluLayer
% 
%     %扩充完成，开始正常提取特征
%     convolution2dLayer([3, 3], 3,'Padding',1)     
%     batchNormalizationLayer   
%     reluLayer
% 
%     convolution2dLayer([3, 3], 3)     
%     batchNormalizationLayer   
%     reluLayer
% 
%     convolution2dLayer([3, 3], 3)     
%     batchNormalizationLayer   
%     reluLayer
% 
%     convolution2dLayer([1, 1], 1)     
% 
%     regressionLayer("Name","output")];                               % 输出层，N*2估计H交替长度  
% 
% 
% options = trainingOptions('adam', ...       % 求解器，'sgdm'（默认） | 'rmsprop' | 'adam'
%     'ExecutionEnvironment','auto', ...
%     'MiniBatchSize', 500, ...                % 批大小,每次训练样本个数90
%     'MaxEpochs',100, ...                   % 最大迭代次数
%     'InitialLearnRate', 0.0005, ...          % 初始化学习速率
%     'LearnRateSchedule','piecewise', ...    % 是否在一定迭代次数后学习速率下降
%     'LearnRateDropFactor',0.95, ...          % 学习速率下降因子
%     'LearnRateDropPeriod',500, ...
%     'Shuffle','every-epoch', ...            % 每次训练打乱数据集
%     'Plots','training-progress',...         % 画出曲线
%     'Verbose',true);                       % 显示训练过程
% 
% 
% net = trainNetwork(XTrainMapD,HTrainMapD,layers,options);
% 
% % 显示网络结构
% analyzeNetwork(net)

load('CNN.mat');
% 训练集、测试集预测
H_sim_1 = predict(net,XTrainMapD);
H_sim_2 = predict(net,XTestMapD);

H_sim_1 = reshape(H_sim_1,[160,45000 * length_snr]);
H_sim_2 = reshape(H_sim_2,[160,5000 * length_snr]);

% 均方根误差
error1 = sqrt(sum((H_sim_1 - H_train).^2) ./ M_size);
error2 = sqrt(sum((H_sim_2 - H_test ).^2) ./ N_size);

figure(1)
subplot(2,1,1)
plot(error1)
xlabel('序列点数')
ylabel('训练集预测均方根误差')
string = {'训练集预测结果对比'};
title(string)
grid

subplot(2,1,2);
plot(error2);
xlabel('序列点数')
ylabel('测试集预测均方根误差')
string = {'测试集预测结果对比'};
title(string)
grid

figure(2)
H_data_3 = H_data_2';
flag_data_3 = flag_data_2';
error3 = sqrt(sum((H_data_3 - flag_data_3 ).^2) ./ N_size);
plot(error3);
xlabel('序列点数')
ylabel('LS估计均方根误差')
string = {'LS估计结果对比'};
title(string)
grid
