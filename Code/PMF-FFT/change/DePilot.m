function [freq_data_SIG,freq_data_PHR, freq_pilot_syms] = DePilot(fre_data_and_pilot,SIG_num,PHR_num)

symbol_num = SIG_num + PHR_num;

IFFT_NUM = 32;
pilot_set_num=3;   %有7组导频
pilot_loc=[7 19;
           15 27;
           11 23];
 data_loc=[8:16 18 20:27;
           7:14 16 18:26;
           7:10 12:16 18:22 24:27];
%     para_time_signal = remove_cp(time_signal,symbol_num);   
%     para_time_signal=reshape(para_time_signal,IFFT_NUM,length(para_time_signal)/(IFFT_NUM)); %串行信号变并行
%     fre_data_and_pilot=fft(para_time_signal); %fft
    
    %-测试- 判断导频位置和载波位置是否出错      
    sum_loc = [pilot_loc,data_loc];
    sum_loc1 = sort(sum_loc.');
%     figure;stem(sum_loc1(:))
    
    fre_data=zeros(size(data_loc,2),size(fre_data_and_pilot,2));   %频域数据
    fre_pilot=zeros(size(pilot_loc,2),size(fre_data_and_pilot,2));  %频域导频
    %-测试- 判断导频位置和载波位置是否出错      
    
    for i=1:size(fre_data_and_pilot,2)
        %先判断使用的是第几组导频
        pilot_set=mod(i,pilot_set_num);
        if pilot_set==0
            if pilot_set_num==7
                pilot_set=7;
            else
                pilot_set=3;
            end
        end
          
        
        data_loc_now=data_loc(pilot_set,:);
        fre_data(:,i)=fre_data_and_pilot(data_loc_now,i);
        
        pilot_loc_now=pilot_loc(pilot_set,:);
        fre_pilot(:,i)=fre_data_and_pilot(pilot_loc_now,i);
    end
    
    freq_data_syms=fre_data;
    freq_pilot_syms=fre_pilot;
      
       
     freq_data_SIG=freq_data_syms(:,1:SIG_num).';
    freq_data_PHR=freq_data_syms(:,SIG_num+1:SIG_num+PHR_num).';
    
    
end