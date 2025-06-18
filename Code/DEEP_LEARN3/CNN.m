clc;
clear all;

num = 10000;       %数据数量  
N = 20;          %子载波数量
length_snr = 5;
H_s_r = xlsread('F:\Project\MATLAB\Deeplearning\FSR_NET\数据集\H_s_1_r.xls');
H_s_i = xlsread('F:\Project\MATLAB\Deeplearning\FSR_NET\数据集\H_s_1_i.xls');
H_p_r = xlsread('F:\Project\MATLAB\Deeplearning\FSR_NET\数据集\H_p_1_r.xls');
H_p_i = xlsread('F:\Project\MATLAB\Deeplearning\FSR_NET\数据集\H_p_1_i.xls');

%数据拼接
trainRatio = 0.9;   %设置 90% 数据用于训练，剩余用于测试

H_data_1 = zeros(trainRatio * num * length_snr,N * 2);
H_data_2 = zeros((num - trainRatio * num) * length_snr,N * 2);
flag_data_1 = zeros(trainRatio * num * length_snr,N * 2);
flag_data_2 = zeros((num - trainRatio * num) * length_snr,N * 2);

for j = 1 : 1 : length_snr
    for i = 1 : 1 : trainRatio * num
        for k = 1 : 1 : N
            I = i + (j - 1) * trainRatio * num;
            J = i + (j - 1) * num;
            H_data_1(I,2 * k - 1) = H_p_r(J,k);
            H_data_1(I,2 * k) = H_p_i(J,k);
            flag_data_1(I,2 * k - 1) = H_s_r(J,k);
            flag_data_1(I,2 * k) = H_s_i(J,k);
        end
    end
end

for j = 1 : 1 : length_snr
    for i = (trainRatio * num + 1) : 1 : num
        for k = 1 : 1 : N
            I = i - trainRatio * num + (j - 1) * (num - trainRatio * num);
            J = i + (j - 1) * num;
            H_data_2(I,2 * k - 1) = H_p_r(J,k);
            H_data_2(I,2 * k) = H_p_i(J,k);
            flag_data_2(I,2 * k - 1) = H_s_r(J,k);
            flag_data_2(I,2 * k) = H_s_i(J,k);
        end
    end
end

%训练测试数据分割
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

%数据转换
XTrainMapD=reshape(X_train,[size(X_train,1) / 8,8,1,M_size]);%训练集输入
XTestMapD =reshape(X_test,[size(X_test,1) / 8,8,1,N_size]); %测试集输入

YTrainMapD=reshape(Y_train,[size(Y_train,1) / 8,8,1,M_size]);%训练集输入
YTestMapD =reshape(Y_test,[size(Y_test,1) / 8,8,1,N_size]); %测试集输入

%神经网络构建
layers = [
    imageInputLayer([size(X_train,1) / 8, 8, 1])    % 输入层，输入长度为2*N

    convolution2dLayer([3, 3], 3,'Padding',2)     % 卷积层，4为滤波器宽度，1为滤波器数量
    batchNormalizationLayer   
    reluLayer                            % relu激活函数

    convolution2dLayer([3, 3], 3,'Padding',2)     % 卷积层，16为滤波器宽度，5为滤波器数量
    batchNormalizationLayer   
    reluLayer                            % relu激活函数

    convolution2dLayer([3, 3], 3,'Padding',2)     % 卷积层，16为滤波器宽度，5为滤波器数量
    batchNormalizationLayer   
    reluLayer

    convolution2dLayer([3, 3], 3,'Padding',2)     % 卷积层，16为滤波器宽度，5为滤波器数量
    batchNormalizationLayer   
    reluLayer                            % relu激活函数

    convolution2dLayer([3, 3], 3,'Padding',2)     % 卷积层，16为滤波器宽度，5为滤波器数量
    batchNormalizationLayer   
    reluLayer                            % relu激活函数


    convolution2dLayer([3, 3], 3,'Padding',1)     % 卷积层，16为滤波器宽度，5为滤波器数量
    batchNormalizationLayer   
    reluLayer                            % relu激活函数

    convolution2dLayer([3, 3], 3,'Padding',1)     % 卷积层，16为滤波器宽度，5为滤波器数量
    batchNormalizationLayer   
    reluLayer                            % relu激活函数

    convolution2dLayer([3, 3], 3,'Padding',1)     % 卷积层，16为滤波器宽度，5为滤波器数量
    batchNormalizationLayer   
    reluLayer                            % relu激活函数

    convolution2dLayer([3, 3], 3)     % 卷积层，16为滤波器宽度，5为滤波器数量
    batchNormalizationLayer   
    reluLayer                            % relu激活函数

    convolution2dLayer([3, 3], 3)     % 卷积层，16为滤波器宽度，5为滤波器数量
    batchNormalizationLayer   
    reluLayer                            % relu激活函数


    convolution2dLayer([1, 1], 2)      % 另一个卷积层
    batchNormalizationLayer 
    reluLayer

    convolution2dLayer([7, 7], 1)      % 另一个卷积层 

    regressionLayer("Name","output")];                               % 输出层，N*2估计H交替长度  


options = trainingOptions('adam', ...       % 求解器，'sgdm'（默认） | 'rmsprop' | 'adam'
    'ExecutionEnvironment','auto', ...
    'MiniBatchSize', 500, ...                % 批大小,每次训练样本个数90
    'MaxEpochs',100, ...                   % 最大迭代次数
    'InitialLearnRate', 0.0005, ...          % 初始化学习速率
    'LearnRateSchedule','piecewise', ...    % 是否在一定迭代次数后学习速率下降
    'LearnRateDropFactor',0.90, ...          % 学习速率下降因子
    'LearnRateDropPeriod',500, ...
    'Shuffle','every-epoch', ...            % 每次训练打乱数据集
    'Plots','training-progress',...         % 画出曲线
    'Verbose',true);                       % 显示训练过程


net = trainNetwork(XTrainMapD,YTrainMapD,layers,options);

% 显示网络结构
analyzeNetwork(net)

% 训练集、测试集预测
Y_sim_1 = predict(net,XTrainMapD);
Y_sim_2 = predict(net,XTestMapD);

Y_sim_1 = reshape(Y_sim_1,[128,9000 * length_snr]);
Y_sim_2 = reshape(Y_sim_2,[128,1000 * length_snr]);

% 均方根误差
error = zeros(45000,1);
for i = 1 : 1 : 45000
    X = 0;
    for j = 1 : 1 : 128
        X = X + (Y_sim_1(j,i) - Y_train(j,i))^2;
    end
    error(i) = X / 128 / var(Y_train(:,j)) ;
end
plot(error);

error = zeros(45000,1);
for i = 1 : 1 : 45000
    X = 0;
    for j = 1 : 1 : 128
        X = X + (Y_sim_1(j,i) - Y_train(j,i))^2;
    end
    error(i) = X / 128 / var(Y_train(:,j)) ;
end
plot(error);


error1 = sqrt(sum((Y_sim_1 - Y_train).^2) ./ 128);
error2 = sqrt(sum((Y_sim_2 - Y_test ).^2) ./ 128);

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
error3 = sqrt(sum((H_data_3 - flag_data_3 ).^2) ./ 128);
plot(error3);
xlabel('序列点数')
ylabel('LS估计均方根误差')
string = {'LS估计结果对比'};
title(string)
grid
