function [LTF] = ltf_gen(CarrierNUM,N_FFT)

    NP_CP_LEN = N_FFT/2;
    
    r = 5;  %指定寄存器位数，也就确定了m序列的长度
    g = 1*ones(1,r);
    PN_out = zeros(1,2^r-1);
    for k = 1 :(2^r-1)
        PN_out(k) = g(r);%%%out
        tmp = xor(g(3),g(r));%将3 10 两位进行异或运算
        g(2:r) = g(1:r-1);
        g(1) = tmp;%将寄存器的最后一位放到第一位，进行下一次移位
    end
    
    P = PN_out(1:CarrierNUM);
    P_BPSK = bpsk_map(P);
    P3 = [0 P_BPSK(CarrierNUM/2+1:CarrierNUM) zeros(1,N_FFT-CarrierNUM-1) P_BPSK(1:CarrierNUM/2)];
    base3 = ifft(P3,N_FFT).*sqrt(N_FFT);
    
    LTF_out = [base3(N_FFT-NP_CP_LEN+1:end),base3(1,:),base3(1,:),base3(1,:),base3(1,:)];
    LTF = 10^(2/20)*LTF_out;

end
