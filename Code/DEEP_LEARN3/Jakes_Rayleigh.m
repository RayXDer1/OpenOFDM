function h = Jakes_Rayleigh(fd, M, dt, N)
T = N*dt - dt;
t = 0:dt:T;
const = sqrt(2/M);                                       % 归一化功率系数
w = 2*pi*fd;
x = 0;
y = 0;
for n = 1:M
    if(n == 1) %主径参数
        alpha = pi/2;      % 第n条入射波的入射角 
        % ph1 = 2*pi*rand - pi;                            % 随机相位服从(-pi,pi)之间的均匀分布
        % ph2 = 2*pi*rand - pi;
        x = x + 2 * (M - n + 1) * const*cos(w*t*cos(alpha));
        y = y + 2 * (M - n + 1) * const*cos(w*t*sin(alpha));
    else
        alpha = (2*pi*n-pi+(2*pi*rand-pi)) / (4*M);      % 第n条入射波的入射角 
        ph1 = 2*pi*rand - pi;                            % 随机相位服从(-pi,pi)之间的均匀分布
        ph2 = 2*pi*rand - pi;
        x = x + 2 * (M - n + 1) * const*cos(w*t*cos(alpha) + ph1);
        y = y + 2 * (M - n + 1) * const*cos(w*t*sin(alpha) + ph2);
        % x = x;
        % y = y;
    end
end

h = (x + 1j*y) / sqrt(2);
