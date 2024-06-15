function [Train_out, Train_Info] = Train_DNN_V1(Train_data, Train_label, Valid_data, Valid_label)

    % Option settings
    ValidationFrequency = 10; %验证频率

        % 预处理数据集 剔除数据集中存在异常点的数据
        TOTAL_NUM = 60000;
        j = 1;
        k = 1;
        for i = 1:TOTAL_NUM
            tempi = Train_label(:,:,:,i);
            tempis = tempi(:).';
            if ((max(abs(tempis))) <= 10.5)
                train_data(:,:,:,j) = Train_data(:,:,:,i);
                train_label(:,:,:,j) = Train_label(:,:,:,i);
                j = j + 1;
            end
        end
        for n = 1:TOTAL_NUM
            tempn = Valid_label(:,:,:,n);
            tempns = tempn(:).';
            if ((max(abs(tempns))) <= 10.5)
                valid_data(:,:,:,k) = Valid_data(:,:,:,n);
                valid_label(:,:,:,k) = Valid_label(:,:,:,n);
                k = k + 1;
            end
        end

        data =  cat(4,train_data,valid_data);
        label =  cat(4,train_label,valid_label);
        
        % 在训练前打乱数据集
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

        % 搭建DNN网络 输入输出大小为64*1
        Layers = [
        imageInputLayer([64 1 1],"Name","imageinput","Normalization","none")
        fullyConnectedLayer(320,"Name","fc_1")
        reluLayer("Name","relu_1")
        fullyConnectedLayer(160,"Name","fc_2")
        reluLayer("Name","relu_2")
        fullyConnectedLayer(120,"Name","fc_3")
        reluLayer("Name","relu_3")
        fullyConnectedLayer(64,"Name","fc_4")
        regressionLayer("Name","regressionoutput")];


    % 设置训练参数
    % MaxEpochs 最大轮次：100
    % MiniBatchSize 最小批量大小：64
    % InitialLearnRate 学习率初始化大小：0.0001
    % LearnRateDropFactor 学习率衰减因子：0.95
    % LearnRateDropPeriod 学习率衰减周期：5轮
    % ValidationData 验证数据集
    % ValidationFrequency 验证频率
    % Shuffle 数据清洗方式 every-epoch 每训练1轮前就打乱1次数据
    Options = trainingOptions('rmsprop', ...
        'MaxEpochs',100, ...
        'MiniBatchSize',64, ...
        'InitialLearnRate',1e-4, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.95, ...
        'LearnRateDropPeriod',5, ...
        'ValidationData',{valid_datas1, valid_labels1}, ...
        'ValidationFrequency',ValidationFrequency, ...
        'Shuffle','every-epoch', ...
        'Verbose',1, ...
        'L2Regularization',0.0005, ...
        'Plots','training-progress');
    
    % 开始训练
    % Train_out：训练完成的含参数的网络
    % Train_Info：训练完成的信息
    [Train_out, Train_Info] = trainNetwork(train_datas1, train_labels1, Layers, Options);



end
