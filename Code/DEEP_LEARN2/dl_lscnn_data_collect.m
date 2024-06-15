function [data,label] = dl_lscnn_data_collect(data_in,ndata_in)

    [a,b] = size(data_in);

    montage_data = [real(data_in);imag(data_in)];
    montage_ndata = [real(ndata_in);imag(ndata_in)];

    montage_datas = montage_data(:);
    montage_ndatas = montage_ndata(:);

    % 数据归一化
    montage_data_norm = 2*(montage_datas - min(montage_datas))./(max(montage_datas)-min(montage_datas))-1;
    montage_ndata_norm = 2*(montage_ndatas - min(montage_datas))./(max(montage_datas)-min(montage_datas))-1;
    % montage_data_norm = montage_data ./ max(abs(montage_data));
    % montage_ndata_norm = montage_ndata ./ max(abs(montage_data));

    % montage_data_norm = montage_datas;
    % montage_ndata_norm = montage_ndatas;
    
    % 将其转化为2*30的图像 转化为图像去噪的思想解决问题
    reshape_montage_data_norm = reshape(montage_data_norm,2,30);
    reshape_montage_ndata_norm = reshape(montage_ndata_norm,2,30);

    data = reshape_montage_data_norm;
    label = reshape_montage_ndata_norm;


end
