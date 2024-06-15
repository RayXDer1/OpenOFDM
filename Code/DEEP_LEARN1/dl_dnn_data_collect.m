function [Train_data_dnn,Train_label_dnn,Valid_data_dnn,Valid_label_dnn] = dl_dnn_data_collect(Train_data_dnn,Train_label_dnn,Valid_data_dnn,Valid_label_dnn,data_in,ndata_in,frame_cnt,total_numbers,Training_set_ratio)

    [a,b] = size(data_in);
    index = randi(a);

    montage_data = [real(data_in(index,:)) imag(data_in(index,:))];
    montage_ndata = [real(ndata_in(index,:)) imag(ndata_in(index,:))];

    % 数据归一化
    montage_data_norm = 2*(montage_data - min(montage_data))./(max(montage_data)-min(montage_data))-1;
    montage_ndata_norm = 2*(montage_ndata - min(montage_data))./(max(montage_data)-min(montage_data))-1;

    % montage_ndata_norm = montage_ndata;

    data = montage_data_norm;
    label = montage_ndata_norm;

    % 数据收集
    if frame_cnt <= fix(Training_set_ratio * total_numbers)
        Train_data_dnn(:, :, :, frame_cnt) = data;
        Train_label_dnn(:, :, :, frame_cnt) = label;
    else
        Valid_data_dnn(:, :, :, frame_cnt - Training_set_ratio * total_numbers) = data;
        Valid_label_dnn(:, :, :, frame_cnt - Training_set_ratio * total_numbers) = label;
    end

end
