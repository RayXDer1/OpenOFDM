clc;
clear all;

num = 10000;       %数据数量  
N = 64;          %子载波数量

H_s_r = xlsread('F:\Project\MATLAB\Deeplearning\FSR_NET\H_s_r.xls');
H_s_i = xlsread('F:\Project\MATLAB\Deeplearning\FSR_NET\H_s_i.xls');
H_p_r = xlsread('F:\Project\MATLAB\Deeplearning\FSR_NET\H_p_r.xls');
H_p_i = xlsread('F:\Project\MATLAB\Deeplearning\FSR_NET\H_p_i.xls');
Y_p_r = xlsread('F:\Project\MATLAB\Deeplearning\FSR_NET\Y_p_r.xls');
Y_p_i = xlsread('F:\Project\MATLAB\Deeplearning\FSR_NET\Y_p_i.xls');
X_p_r = xlsread('F:\Project\MATLAB\Deeplearning\FSR_NET\X_p_r.xls');
X_p_i = xlsread('F:\Project\MATLAB\Deeplearning\FSR_NET\X_p_i.xls');

%数据拼接
for i = 1 : 1 : num
    for k = 1 : 1 : N
        X_data(i,2 * k - 1) = X_p_r(i,k);
        X_data(i,2 * k) = X_p_i(i,k);
        Y_data(i,2 * k - 1) = Y_p_r(i,k);
        Y_data(i,2 * k) = Y_p_i(i,k);
        H_data(i,2 * k - 1) = H_p_r(i,k);
        H_data(i,2 * k) = H_p_i(i,k);
        flag_data(i,2 * k - 1) = H_s_r(i,k);
        flag_data(i,2 * k) = H_s_i(i,k);
    end
end

for i = 1 : 1 : num
    % data_in(i,:) = [X_data(i,:),Y_data(i,:),H_data(i,:)];  %三路拼接输入
    data_in(i,:) = H_data(i,:);                             %仅用H_LS作为输入
end

%训练测试数据分割
trainRatio = 0.9;   %设置 90% 数据用于训练，剩余用于测试
trainCount = floor(num * trainRatio);
X_train = data_in(1 : trainCount,:);
X_test = data_in(trainCount + 1:num,:);

X_train = X_train';
X_test = X_test';

M_size = size(X_train, 2);
N_size = size(X_test, 2);

Y_train = flag_data(1 : trainCount,:);
Y_test = flag_data(trainCount + 1:num,:);

% Y_train = Y_train';
% Y_test = Y_test';

%数据转换
XTrainMapD=reshape(X_train,[size(X_train,1),1,1,M_size]);%训练集输入
XTestMapD =reshape(X_test,[size(X_test,1),1,1,N_size]); %测试集输入

% YTrainMapD=reshape(Y_train,[length(Y_train),1,1,90]);%训练集输入
%YTestMapD =reshape(Y_test,[1,length(Y_test),1,size(Y_test,1)]); %测试集输入

%神经网络构建
layers = [
    imageInputLayer([size(X_train,1), 1, 1])    % 输入层，输入长度为2*N

    convolution2dLayer([8, 1], 4, 'Padding','same')     % 卷积层，4为滤波器宽度，1为滤波器数量
    batchNormalizationLayer   
    reluLayer                            % relu激活函数

    convolution2dLayer([4, 1], 1, 'Padding','same')     % 卷积层，4为滤波器宽度，1为滤波器数量
    batchNormalizationLayer   
    reluLayer                            % relu激活函数

    % convolution2dLayer([4, 1], 4, 'Padding','same')     % 卷积层，16为滤波器宽度，5为滤波器数量
    % batchNormalizationLayer   
    % reluLayer                            % relu激活函数

    convolution2dLayer([1, 1], 2, 'Padding','same')      % 另一个卷积层
    batchNormalizationLayer 
    reluLayer

    convolution2dLayer([32, 1], 2, 'Padding','same')      % 另一个卷积层
    batchNormalizationLayer 
    reluLayer

    % fullyConnectedLayer(384)
    % reluLayer
    % 
    % fullyConnectedLayer(256)
    % reluLayer
    % 
    % fullyConnectedLayer(256)
    % reluLayer
  
    dropoutLayer(0.2)
    fullyConnectedLayer(128)

    regressionLayer("Name","output")];                               % 输出层，N*2估计H交替长度  


options = trainingOptions('adam', ...       % 求解器，'sgdm'（默认） | 'rmsprop' | 'adam'
    'ExecutionEnvironment','auto', ...
    'MiniBatchSize', 50, ...                % 批大小,每次训练样本个数90
    'MaxEpochs',100, ...                   % 最大迭代次数
    'InitialLearnRate', 0.005, ...          % 初始化学习速率
    'LearnRateSchedule','piecewise', ...    % 是否在一定迭代次数后学习速率下降
    'LearnRateDropFactor',0.9, ...          % 学习速率下降因子
    'LearnRateDropPeriod',500, ...
    'Shuffle','every-epoch', ...            % 每次训练打乱数据集
    'Plots','training-progress',...         % 画出曲线
    'Verbose',true);                       % 显示训练过程


net = trainNetwork(XTrainMapD,Y_train,layers,options);

% 显示网络结构
analyzeNetwork(net)

% 训练集、测试集预测
Y_sim_1 = predict(net,XTrainMapD);
Y_sim_2 = predict(net,XTestMapD);

% 均方根误差
error1 = sqrt(sum((Y_sim_1 - Y_train).^2) ./ M_size);
error2 = sqrt(sum((Y_sim_2 - Y_test ).^2) ./ N_size);

figure(1)
subplot(2,1,1)
plot(error1)
xlabel('序列点数')
ylabel('训练集预测均方根误差')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
grid

subplot(2,1,2);
plot(error2);
xlabel('序列点数')
ylabel('测试集预测均方根误差')
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
grid


