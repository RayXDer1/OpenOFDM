function [PSDU_PB,PSDU_BPC,PSDU_rate] = mode_control(mode)

    %模式控制模块
    
    PSDU_PARA = [16 , 1  , 0.5;     %BPSK
                 16 , 1  , 0.8;     %BPSK
                 16 , 2  , 0.5;     %QPSK
                 16 , 2  , 0.8;     %QPSK
                 16 , 4  , 0.5;     %16QAM
                 16 , 4  , 0.8;     %16QAM
                 72 , 4  , 0.5;     %16QAM
                 72 , 4  , 0.8;     %16QAM
                 136, 4  , 0.5;     %16QAM
                 136, 4  , 0.8;     %16QAM
                 520, 1  , 0.5;     %BPSK 
                 520, 2  , 0.5;     %QPSK
                 520, 4  , 0.5;     %16QAM
                 520, 6  , 0.5;     %64QAM
                 520, 8  , 0.5;     %256QAM
                 520, 10 , 0.5;     %1024QAM
                 4  , 1  , 0.5;     %BPSK 验证polar
                 48 , 2  , 0.5];    %QPSK 

    PSDU_PB   = PSDU_PARA(mode+1,1);  %字节长度
    PSDU_BPC  = PSDU_PARA(mode+1,2);  %调制方式
    PSDU_rate = PSDU_PARA(mode+1,3);  %编码码率

end