function data_out = data_gen(PSDU_PB)
%随机数据生成模块

    PSDU_LENGTH = PSDU_PB * 8;
    data_out = round(rand(1,PSDU_LENGTH));

end