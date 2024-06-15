addpath DATA\

% 导入本地存储的数据集
Train_data1 = load("Train_data_cnn.mat").Train_data_cnn;
Train_label1 = load("Train_label_cnn.mat").Train_label_cnn;
Valid_data1 = load("Valid_data_cnn.mat").Valid_data_cnn;
Valid_label1 = load("Valid_label_cnn.mat").Valid_label_cnn;

% 开始进行神经网络训练

% 残差CNN网络
[Train_out_rescnn, Train_Info_rescnn] = Train_LS_RESCNN_V1(Train_data1, Train_label1, Valid_data1, Valid_label1);

% CNN网络
% [Train_out_cnn, Train_Info_cnn] = Train_LSCNN_V1(Train_data1, Train_label1, Valid_data1, Valid_label1);