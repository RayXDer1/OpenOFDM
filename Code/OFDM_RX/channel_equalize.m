function [RX_EQ_OUT] = channel_equalize(rx_data,H_LS_EST)

    RX_EQ_OUT = rx_data ./ H_LS_EST;

end
