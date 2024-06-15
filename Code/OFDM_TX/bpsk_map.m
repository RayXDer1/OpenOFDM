function data_out = bpsk_map(data_in)

    data_out = 2 * (data_in - 0.5);
    data_out(data_out > 1) = 1;
    data_out(data_out < -1) = -1;

end