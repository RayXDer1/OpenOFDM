function [Train_out, Train_Info] = Train_LSCNN_V1(Train_data, Train_label, Valid_data, Valid_label)

    % Option settings
    ValidationFrequency = 10;

        TOTAL_NUM = 60000;
        j = 1;
        k = 1;
        for i = 1:TOTAL_NUM
            tempi = Train_label(:,:,:,i);
            tempis = tempi(:).';
            if ((max(abs(tempis))) <= 1.5)
                train_data(:,:,:,j) = Train_data(:,:,:,i);
                train_label(:,:,:,j) = Train_label(:,:,:,i);
                j = j + 1;
            end
        end
        for n = 1:TOTAL_NUM
            tempn = Valid_label(:,:,:,n);
            tempns = tempn(:).';
            if ((max(abs(tempns))) <= 1.5)
                valid_data(:,:,:,k) = Valid_data(:,:,:,n);
                valid_label(:,:,:,k) = Valid_label(:,:,:,n);
                k = k + 1;
            end
        end

        data =  cat(4,train_data,valid_data);
        label =  cat(4,train_label,valid_label);

        % data =  cat(4,Train_data,Valid_data);
        % label =  cat(4,Train_label,Valid_label);
        
        N_total = length(data);
        ratio = 0.9;
        N_train = fix(N_total * ratio);
        N_test =  N_total - N_train;
        u = randperm(N_total);
        % u = 1:N_total;
        m = 1:N_total;
        P = 1:N_train;
        Q = (N_train+1):N_total;
        data(:,:,:,u) = data(:,:,:,m);
        label(:,:,:,u) = label(:,:,:,m);
        train_datas(:,:,:,P)  = data(:,:,:,P);
        train_labels(:,:,:,P)  = label(:,:,:,P);
        valid_datas(:,:,:,(Q-N_train))  = data(:,:,:,Q);
        valid_labels(:,:,:,(Q-N_train))  = label(:,:,:,Q);

        for p1 = 1:N_train
            train_datas1(:,:,:,p1)  = train_datas(:,:,:,p1);
            train_labels1(:,:,:,p1) = train_labels(:,:,:,p1);
        end
        for p2 = 1:N_test
            valid_datas1(:,:,:,p2)  = valid_datas(:,:,:,p2);
            valid_labels1(:,:,:,p2) = valid_labels(:,:,:,p2);
        end
        

       channel_num = 16;

        Layers = [ ...
        imageInputLayer([2 30 1],'Normalization','none')
        % 使用4个卷积核分别与对输入的1通道数据进行卷积
        % 卷积结果求和加上4个偏置，最后输出4个feature map数据
        convolution2dLayer(3,channel_num,'Padding',[2 2])
        reluLayer
        % 使用4个各自的卷积核分别与对输入的4通道各自的数据进行卷积
        % 卷积结果求和加上4个偏置，最后输出4个feature map数据
        convolution2dLayer(3,channel_num,'Padding',[2 2],'NumChannels',channel_num)
        reluLayer
        convolution2dLayer(3,channel_num,'Padding',[2 2],'NumChannels',channel_num)
        reluLayer
        convolution2dLayer(3,channel_num,'Padding',[2 2],'NumChannels',channel_num)
        reluLayer
        convolution2dLayer(3,channel_num,'Padding',[1 1],'NumChannels',channel_num)
        reluLayer
        convolution2dLayer(3,channel_num,'Padding',[1 1],'NumChannels',channel_num)
        reluLayer
        convolution2dLayer(3,channel_num,'Padding',[1 1],'NumChannels',channel_num)
        reluLayer
        convolution2dLayer(3,channel_num,'Padding',[1 1],'NumChannels',channel_num)
        reluLayer 
        convolution2dLayer(3,channel_num,'Padding',[0 0],'NumChannels',channel_num)
        reluLayer 
        convolution2dLayer(3,channel_num,'Padding',[0 0],'NumChannels',channel_num)
        reluLayer 
        convolution2dLayer(3,channel_num,'Padding',[0 0],'NumChannels',channel_num)
        reluLayer       
        convolution2dLayer(3,1,'Padding',0,'NumChannels',channel_num)
        regressionLayer
        ];

    feature_num = 16;
    llayers = [
    imageInputLayer([2 30 1],'Name','input')
    
    convolution2dLayer(3,feature_num,'Padding','same','Name','conv_1')
    % batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu1')
    
    % Residual Block 1
    convolution2dLayer(3,feature_num,'Padding','same','Name','res1_conv1')
    % batchNormalizationLayer('Name','res1_BN1')
    reluLayer('Name','res1_relu1')
    convolution2dLayer(3,feature_num,'Padding','same','Name','res1_conv2')
    % batchNormalizationLayer('Name','res1_BN2')
    additionLayer(2,'Name','res1_add')
    reluLayer('Name','res1_relu2')
    
    % Residual Block 2
    convolution2dLayer(3,feature_num,'Padding','same','Name','res2_conv1')
    % batchNormalizationLayer('Name','res2_BN1')
    reluLayer('Name','res2_relu1')
    convolution2dLayer(3,feature_num,'Padding','same','Name','res2_conv2')
    % batchNormalizationLayer('Name','res2_BN2')
    additionLayer(2,'Name','res2_add')
    reluLayer('Name','res2_relu2')

    % Residual Block 3
    convolution2dLayer(3, feature_num, 'Padding', 'same', 'Name', 'res3_conv1')
    % batchNormalizationLayer('Name', 'res3_batchnorm1')
    reluLayer('Name', 'res3_relu1')
    convolution2dLayer(3, feature_num, 'Padding', 'same', 'Name', 'res3_conv2')
    % batchNormalizationLayer('Name', 'res3_batchnorm2')
    additionLayer(2, 'Name', 'res3_add')
    reluLayer('Name', 'res3_relu2')

    % Residual Block 4
    convolution2dLayer(3, feature_num, 'Padding', 'same', 'Name', 'res4_conv1')
    % batchNormalizationLayer('Name', 'res4_batchnorm1')
    reluLayer('Name', 'res4_relu1')
    convolution2dLayer(3, feature_num, 'Padding', 'same', 'Name', 'res4_conv2')
    % batchNormalizationLayer('Name', 'res4_batchnorm2')
    additionLayer(2, 'Name', 'res4_add')
    reluLayer('Name', 'res4_relu2')

    % % Residual Block 5
    % convolution2dLayer(3, feature_num, 'Padding', 'same', 'Name', 'res5_conv1')
    % % batchNormalizationLayer('Name', 'res5_batchnorm1')
    % reluLayer('Name', 'res5_relu1')
    % convolution2dLayer(3, feature_num, 'Padding', 'same', 'Name', 'res5_conv2')
    % % batchNormalizationLayer('Name', 'res5_batchnorm2')
    % additionLayer(2, 'Name', 'res5_add')
    % reluLayer('Name', 'res5_relu2')
    % 
    % % Residual Block 6
    % convolution2dLayer(3, feature_num, 'Padding', 'same', 'Name', 'res6_conv1')
    % % batchNormalizationLayer('Name', 'res6_batchnorm1')
    % reluLayer('Name', 'res6_relu1')
    % convolution2dLayer(3, feature_num, 'Padding', 'same', 'Name', 'res6_conv2')
    % % batchNormalizationLayer('Name', 'res6_batchnorm2')
    % additionLayer(2, 'Name', 'res6_add')
    % reluLayer('Name', 'res6_relu2')
    % 
    % % Residual Block 7
    % convolution2dLayer(3, feature_num, 'Padding', 'same', 'Name', 'res7_conv1')
    % % batchNormalizationLayer('Name', 'res7_batchnorm1')
    % reluLayer('Name', 'res7_relu1')
    % convolution2dLayer(3, feature_num, 'Padding', 'same', 'Name', 'res7_conv2')
    % % batchNormalizationLayer('Name', 'res7_batchnorm2')
    % additionLayer(2, 'Name', 'res7_add')
    % reluLayer('Name', 'res7_relu2')
    % 
    % % Residual Block 8
    % convolution2dLayer(3, feature_num, 'Padding', 'same', 'Name', 'res8_conv1')
    % % batchNormalizationLayer('Name', 'res8_batchnorm1')
    % reluLayer('Name', 'res8_relu1')
    % convolution2dLayer(3, feature_num, 'Padding', 'same', 'Name', 'res8_conv2')
    % % batchNormalizationLayer('Name', 'res8_batchnorm2')
    % additionLayer(2, 'Name', 'res8_add')
    % reluLayer('Name', 'res8_relu2')
    
    convolution2dLayer(3,1,'Padding','same','Name','conv_2')
    
    regressionLayer('Name','output')
    ];

lgraph = layerGraph(llayers);

% Connect residual blocks
lgraph = connectLayers(lgraph,'relu1','res1_add/in2');
lgraph = connectLayers(lgraph,'res1_relu2','res2_add/in2');
lgraph = connectLayers(lgraph, 'res2_relu2', 'res3_add/in2');
lgraph = connectLayers(lgraph, 'res3_relu2', 'res4_add/in2');
% lgraph = connectLayers(lgraph, 'res4_relu2', 'res5_add/in2');
% lgraph = connectLayers(lgraph, 'res5_relu2', 'res6_add/in2');
% lgraph = connectLayers(lgraph, 'res6_relu2', 'res7_add/in2');
% lgraph = connectLayers(lgraph, 'res7_relu2', 'res8_add/in2');

% plot(lgraph);



    Options = trainingOptions('rmsprop', ...
        'MaxEpochs',50, ...
        'MiniBatchSize',256, ...
        'InitialLearnRate',2e-5, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.96, ...
        'LearnRateDropPeriod',1, ...
        'ValidationData',{valid_datas1, valid_labels1}, ...
        'ValidationFrequency',ValidationFrequency, ...
        'Shuffle','every-epoch', ...
        'Verbose',1, ...
        'L2Regularization',0.0005, ...
        'Plots','training-progress');
    
    [Train_out, Train_Info] = trainNetwork(train_datas1, train_labels1, Layers, Options);



end
