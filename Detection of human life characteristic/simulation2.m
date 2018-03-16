clc;
clear all;

fs=50;
t=0:1/fs:1023/fs; %采样间隔为0.02s,即采样频率为50Hz
N=length(t);
noise=randn(1,N);
I=cos(1.764*sin(0.7*pi*t+pi/6)+0.22*sin(2.24*pi*t+pi/3)+1.5*pi)+noise;  %仿真的I路信号
Q=sin(1.764*sin(0.7*pi*t+pi/6)+0.22*sin(2.24*pi*t+pi/3)+1.5*pi)+noise;  %仿真的Q路信号

%% 画出原始IQ时域图
figure(1)
subplot(211)
plot(t,I)
ylabel('I[n]')
subplot(212)
plot(t,Q)
ylabel('Q[n]')


%% 1、使用CSD解调
complex_signal=I+1j*Q;
complex_spectrum=fft(complex_signal);
fI=(0:N-1)*fs/N;  %频谱点对应的频率
figure(2)
%plot(fI(1:N/2),complex_spectrum(1:N/2)); %频谱前半段和后半段是对称的，所以只看一半
plot(fI(5:round(3000/fs)),abs(complex_spectrum(5:round(3000/fs))));   %只看前3Hz的频段
title('CSD Algorithm');

%% 2、使用DACM解调
Accum_deriva=zeros(1,N-1);
for i=2:N
    Accum_deriva(i-1)=(I(i)*(Q(i)-Q(i-1))-Q(i)*(I(i)-I(i-1)))/(I(i)^2+Q(i)^2);
end
phase_arc=zeros(1,N-1);    %DACM解调的体征混合信号
for i=1:N-1
    phase_arc(i)=sum(Accum_deriva(1:i));
end
arc_spectrum=fft(phase_arc);
fI=(0:N-2)*fs/(N-1);  %频谱点对应的频率
figure(3)
%plot(fI(1:N/2),complex_spectrum(1:N/2)); %频谱前半段和后半段是对称的，所以只看一半
plot(fI(5:round(3000/fs)),abs(arc_spectrum(5:round(3000/fs))));   %只看前3Hz的频段
title('DACM Algorithm')

%% 3、使用参数化解调，粒子群优化算法
complex_signal=I+1j*Q;
x_range=[-5,5;0.15,0.6;0,2*pi];
Pg=Vital_particleSwarm(complex_signal,t,x_range);

% 输出最终计算结果
disp('函数的全局最优位置为：')
fprintf('当前最优点为：(%s,%s,%s)\n',Pg(1),Pg(2),Pg(3));

%{
c1=1.4962;
c2=1.4962;
w=0.7298;
maxDt=100;    %最大迭代次数
D=3;           %未知量个数
N=40;          %粒子群体个数
eps=10^(-7);   %迭代精度

% 初始化种群的个体
x=randn(N,D);  %初始化种群位置
v=randn(N,D);  %初始化种群速度

% 先计算各个初始粒子的适应度，确定个体初始最优位置y和全局最优位置Pg
p=zeros(1,N);
y=zeros(N,D);
for i=1:N
    p(i)=sphere(complex_signal,t,x(i,:));  %计算适应度，优化函数为sphere
    y(i,:)=x(i,:);          %个体最优位置为t=0时候的位置
end
[max_value,max_pos]=max(p);
Pg=x(max_pos,:);            %全局最优位置
Pg_best=max_value;       %全局最优值

% 训练迭代，找到最优位置，直到满足精度要求或者迭代次数完成
for iter=1:maxDt
    for i=1:N
        v(i,:)=w*v(i,:)+c1*rand(1)*(y(i,:)-x(i,:))+c2*rand(1)*(Pg-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        px=sphere(complex_signal,t,x(i,:));
        if px>p(i)
            p(i)=px;    %更新适应度
            y(i,:)=x(i,:);  %更新个体最佳位置
        end
        if p(i)>Pg_best
            Pg=y(i,:);
            Pg_best=p(i);
        end
    end
    if mod(iter,10)==0
        disp('**************************************');
        fprintf('当前第(%d)次迭代最优点为：(%s,%s,%s)\n',iter,Pg(1),Pg(2),Pg(3));
    end
end

% 输出最终计算结果
disp('函数的全局最优位置为：')
fprintf('当前最优点为：(%s,%s,%s)\n',Pg(1),Pg(2),Pg(3));
%}

%% 3.1 第二次使用粒子群算法，消除呼吸分量二次谐波
disp('****************************************************************')
disp('***************消除呼吸分量后，消除两次呼吸谐波***********************')

Sigma_t=exp(-1j*Pg(1)*sin(2*pi*Pg(2)*t+Pg(3)));
complex_signal2=complex_signal.*Sigma_t;
x_range1=[-5,5;2*Pg(2)-0.034,2*Pg(2)+0.034;0,2*pi]; %单次谐波的取值范围
Pg1=Vital_particleSwarm(complex_signal2,t,x_range1);

% 输出最终计算结果
disp('函数的全局最优位置为：')
fprintf('当前最优点为：(%s,%s,%s)\n',Pg1(1),Pg1(2),Pg1(3));

%{
c1=1.4962;
c2=1.4962;
w=0.7298;
maxDt=100;    %最大迭代次数
D=3;           %未知量个数
N=40;          %粒子群体个数
eps=10^(-7);   %迭代精度

% 初始化种群的个体
x=randn(N,D);  %初始化种群位置
v=randn(N,D);  %初始化种群速度

% 先计算各个初始粒子的适应度，确定个体初始最优位置y和全局最优位置Pg
p=zeros(1,N);
y=zeros(N,D);
for i=1:N
    p(i)=sphere(complex_signal2,t,x(i,:));  %计算适应度，优化函数为sphere
    y(i,:)=x(i,:);          %个体最优位置为t=0时候的位置
end
[max_value,max_pos]=max(p);
Pg1=x(max_pos,:);            %全局最优位置
Pg_best=max_value;       %全局最优值

% 训练迭代，找到最优位置，直到满足精度要求或者迭代次数完成
for iter=1:maxDt
    for i=1:N
        v(i,:)=w*v(i,:)+c1*rand(1)*(y(i,:)-x(i,:))+c2*rand(1)*(Pg1-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        px=sphere(complex_signal2,t,x(i,:));
        if px>p(i)
            p(i)=px;    %更新适应度
            y(i,:)=x(i,:);  %更新个体最佳位置
        end
        if p(i)>Pg_best
            Pg1=y(i,:);
            Pg_best=p(i);
        end
    end
    if mod(iter,10)==0
        disp('**************************************');
        fprintf('当前第(%d)次迭代最优点为：(%s,%s,%s)\n',iter,Pg1(1),Pg1(2),Pg1(3));
    end
end

% 输出最终计算结果
disp('函数的全局最优位置为：')
fprintf('当前最优点为：(%s,%s,%s)\n',Pg1(1),Pg1(2),Pg1(3));
%}

%% 3.2 提取出呼吸分量后，消除呼吸分量三次谐波
disp('****************************************************************')
disp('***************消除呼吸分量后，消除三次谐波分量***********************')

Sigma_t=exp(-1j*Pg1(1)*sin(2*pi*Pg1(2)*t+Pg1(3)));
complex_signal3=complex_signal2.*Sigma_t;
x_range2=[-5,5;3*Pg(2)-0.051,3*Pg(2)+0.051;0,2*pi];  %二次谐波取值范围
Pg2=Vital_particleSwarm(complex_signal3,t,x_range2);

% 输出最终计算结果
disp('函数的全局最优位置为：')
fprintf('当前最优点为：(%s,%s,%s)\n',Pg2(1),Pg2(2),Pg2(3));

%{
c1=1.4962;
c2=1.4962;
w=0.7298;
maxDt=100;    %最大迭代次数
D=3;           %未知量个数
N=40;          %粒子群体个数
eps=10^(-7);   %迭代精度

% 初始化种群的个体
x=randn(N,D);  %初始化种群位置
v=randn(N,D);  %初始化种群速度

% 先计算各个初始粒子的适应度，确定个体初始最优位置y和全局最优位置Pg
p=zeros(1,N);
y=zeros(N,D);
for i=1:N
    p(i)=sphere(complex_signal3,t,x(i,:));  %计算适应度，优化函数为sphere
    y(i,:)=x(i,:);          %个体最优位置为t=0时候的位置
end
[max_value,max_pos]=max(p);
Pg2=x(max_pos,:);            %全局最优位置
Pg_best=max_value;       %全局最优值

% 训练迭代，找到最优位置，直到满足精度要求或者迭代次数完成
for iter=1:maxDt
    for i=1:N
        v(i,:)=w*v(i,:)+c1*rand(1)*(y(i,:)-x(i,:))+c2*rand(1)*(Pg2-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        px=sphere(complex_signal3,t,x(i,:));
        if px>p(i)
            p(i)=px;    %更新适应度
            y(i,:)=x(i,:);  %更新个体最佳位置
        end
        if p(i)>Pg_best
            Pg2=y(i,:);
            Pg_best=p(i);
        end
    end
    if mod(iter,10)==0
        disp('**************************************');
        fprintf('当前第(%d)次迭代最优点为：(%s,%s,%s)\n',iter,Pg2(1),Pg2(2),Pg2(3));
    end
end

% 输出最终计算结果
disp('函数的全局最优位置为：')
fprintf('当前最优点为：(%s,%s,%s)\n',Pg2(1),Pg2(2),Pg2(3));
%}

%% 3.3 消除呼吸分量三次谐波然后重新寻找心跳分量
disp('****************************************************************')
disp('***************消除呼吸分量和谐波分量后，寻找心跳分量***********************')
    
Sigma_t=exp(-1j*Pg2(1)*sin(2*pi*Pg2(2)*t+Pg2(3)));
complex_signal4=complex_signal3.*Sigma_t;
x_range3=[-5,5;0.8,2;0,2*pi];       %心跳取值范围
Pg3=Vital_particleSwarm(complex_signal4,t,x_range3);

% 输出最终计算结果
disp('函数的全局最优位置为：')
fprintf('当前最优点为：(%s,%s,%s)\n',Pg3(1),Pg3(2),Pg3(3));


%{
c1=1.4962;
c2=1.4962;
w=0.7298;
maxDt=100;    %最大迭代次数
D=3;           %未知量个数
N=40;          %粒子群体个数
eps=10^(-7);   %迭代精度

% 初始化种群的个体
x=randn(N,D);  %初始化种群位置
v=randn(N,D);  %初始化种群速度

% 先计算各个初始粒子的适应度，确定个体初始最优位置y和全局最优位置Pg
p=zeros(1,N);
y=zeros(N,D);
for i=1:N
    p(i)=sphere(complex_signal4,t,x(i,:));  %计算适应度，优化函数为sphere
    y(i,:)=x(i,:);          %个体最优位置为t=0时候的位置
end
[max_value,max_pos]=max(p);
Pg3=x(max_pos,:);            %全局最优位置
Pg_best=max_value;       %全局最优值

% 训练迭代，找到最优位置，直到满足精度要求或者迭代次数完成
for iter=1:maxDt
    for i=1:N
        v(i,:)=w*v(i,:)+c1*rand(1)*(y(i,:)-x(i,:))+c2*rand(1)*(Pg3-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        px=sphere(complex_signal4,t,x(i,:));
        if px>p(i)
            p(i)=px;    %更新适应度
            y(i,:)=x(i,:);  %更新个体最佳位置
        end
        if p(i)>Pg_best
            Pg3=y(i,:);
            Pg_best=p(i);
        end
    end
    if mod(iter,10)==0
        disp('**************************************');
        fprintf('当前第(%d)次迭代最优点为：(%s,%s,%s)\n',iter,Pg3(1),Pg3(2),Pg3(3));
    end
end

% 输出最终计算结果
disp('函数的全局最优位置为：')
fprintf('当前最优点为：(%s,%s,%s)\n',Pg3(1),Pg3(2),Pg3(3));
%}

%% 画出经过几次滤波后的波形
figure(4);
complex_spectrum=abs(fft(complex_signal));
complex_spectrum=complex_spectrum/max(complex_spectrum);
complex_spectrum2=abs(fft(complex_signal2));
complex_spectrum2=2*complex_spectrum2/max(complex_spectrum2);
complex_spectrum3=abs(fft(complex_signal3));
complex_spectrum3=complex_spectrum3/max(complex_spectrum3);
fI=(0:N-1)*fs/N;  %频谱点对应的频率

%plot(fI(1:N/2),complex_spectrum(1:N/2)); %频谱前半段和后半段是对称的，所以只看一半
t1=fI(1:round(3000/fs));
plot(t1,complex_spectrum(1:round(3000/fs)),'b',t1,complex_spectrum2(1:round(3000/fs)) ,'r',t1,complex_spectrum3(1:round(3000/fs)),'g');   %只看前3Hz的频段
legend('原始的信号频谱','消除了呼吸频率','继续消除了两次谐波');
title('多次滤除谐波后的曲线');
