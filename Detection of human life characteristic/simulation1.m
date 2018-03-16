% clc;
% clear all;

%% 模拟心跳和呼吸信号
%心跳频率1.4HZ，幅度0.05；呼吸频率为0.4HZ，幅度为1
A1=0.1;
f1=1.4;
A2=1;
f2=0.4;
fs=20; %采样频率
dt=1/fs; %采样时间间隔
n=1024;  %采样点数
t=dt:dt:n/fs;
s1=A1*sin(2*pi*f1*t);
s2=A2*sin(2*pi*f2*t);
subplot(211);
plot(s1);
y_s1=hilbert(s1);
hold on;
plot(abs(y_s1),'r');
title('心跳时域图');
subplot(212);
plot(s2);
y_s2=hilbert(s2);
hold on;
plot(abs(y_s2),'r');
title('呼吸时域图');

%混合信号
figure;
d=s1+s2;
N=n;         %N-fft所用的数据个数
d_f=fft(d,N);  
subplot(211);
plot(d);
title('混合信号图');
subplot(212);
mag=abs(d_f);
f=(0:N-1)*fs/N;
plot(f(1:N/2),mag(1:N/2)*2/N);
title('混合信号频谱');

%加入噪声后的信号
figure;
fn=5;
fi=randn(1,n);
sn=0.7*cos(2*pi*fn*t); %系统内自抖动干扰
d=s1+s2;
dn=d;  %没有收到干扰的信号

%三种加白噪声的方法
%1
% d=awgn(d,10);
% dn1=d; %仅仅加入噪声的信号

%2
%noise=wgn(1,n,-5);
%d=d+noise;

%3
%加指定信噪比白噪声
SNR=-3;
noise=randn(1,n);
signal_power=(1/n)*sum(d.*d);
noise_variance=signal_power/(10^(SNR/10));
noise=sqrt(noise_variance)/std(noise)*noise;
d=d+noise;
dn1=d;
%end

d=d+sn;   %有噪声和干扰信号
dd=dn+sn; %没有噪声，只有干扰信号
N=n;         %N-fft所用的数据个数
d_f=fft(d,N);  
subplot(211);
plot(d);
title('混合信号+噪声图');
subplot(212);
mag=abs(d_f);
f=(0:N-1)*fs/N;
plot(f(1:N/2),mag(1:N/2)*2/N);
title('混合信号+噪声频谱');

%% LMS进行滤波
%{
% 使用普通LMS算法进行滤波
M1=80;
xn=d';
en = zeros(n,1);             % 误差序列,en(k)表示第k次迭代时预期输出与实际输入的误差
W  = zeros(M1,1);             % 每一行代表一个加权参量,每一列代表-次迭代,初始为0
yn=zeros(size(xn));
%yn(1:M1-1)=xn(1:M1-1);

rho_max=max(eig(xn*xn'));
mu=rand()*(1/rho_max);
%mu=0.001;

%FIR+LMS滤波器
% [b,a]=butter(8,0.25);
% dn=filter(b,a,dd);


% LMS迭代计算
for k = M1:n                  % 第k次迭代
    x = xn(k:-1:k-M1+1);        % 滤波器M个抽头的输入
    yn(k) = W.' * x;        % 滤波器的输出
    en(k) = dn(k) - yn(k) ;        % 第k次迭代的误差
    
    % 滤波器权值计算的迭代式
    W = W + 2*mu*en(k)*x;
    % 归一化LMS
    %W = W + (mu/norm(x))*en(k)*x;
end
%}

%% RLS算法进行滤波
%
M1=30;
xn=dn';
en = zeros(n,1);             % 误差序列,en(k)表示第k次迭代时预期输出与实际输入的误差
W  = zeros(M1,1);             % 每一行代表一个加权参量,每一列代表-次迭代,初始为0
T=eye(M1,M1)*10;
yn=zeros(size(xn));
%yn(1:M1-1)=xn(1:M1-1);

rho_max=max(eig(xn*xn'));
mu=rand()*(1/rho_max);
%mu=0.001;
lambda=0.98; %遗忘因子

%FIR+LMS滤波器
% [b,a]=butter(8,0.25);
% dn=filter(b,a,dd);

for k = M1+1:n                  % 第k次迭代
    x = xn(k-1:-1:k-M1);        % 滤波器M个抽头的输入
    K=(T*x)/(lambda+x'*T*x);
    en(k)=dn(k)-W'*x;       % 第k次迭代的误差
    W=W+K*en(k);
    yn(k) = W.' * x;        % 滤波器的输出      
    T=(T-K*x'*T)/lambda;
end
dlmwrite('W.txt',W);
%}

figure;
% plot(t,yn,'b',t,dn,'g',t,en,'r');
% legend('滤波器输出','预期输出','误差');
subplot(311)
plot(t,yn,'b');
title('滤波器输出');
subplot(312)
plot(t,dn,'g');
title('预期输出');
subplot(313);
plot(t,en,'r');
title('误差');

figure;
yn_f=fft(yn,N);
dn_f=fft(dn,N);
subplot(211)
mag_yn=abs(yn_f);
plot(f(1:N/2),mag_yn(1:N/2));
title('滤波器输出频谱');
subplot(212)
mag_dn=abs(dn_f);
plot(f(1:N/2),mag_dn(1:N/2));
title('预期输出频谱');



%% 动态MTI 滤波（滤除反射杂波）
%{
w0=2.4e3;
fss=100;
dt=1/fss;
nn=1024;
t=0:dt:(nn-1)*dt;
xx=cos(w0*t);
L=0.3;
nx=0.5*cos(w0*t-16*L);
nnx=xx+nx;
figure;
subplot(211);
plot(xx)
title('期望信号时域，频域图');
subplot(212);
xx_f=fft(xx);
mag_xxf=abs(xx_f);
f=(0:nn-1)*fss/nn;
plot(f(1:nn/2),mag_xxf(1:nn/2)*2/nn);
title('混合信号+噪声频谱');

figure;
subplot(211);
plot(nnx)
title('期望信号时域，频域图');
subplot(212);
nnx_f=fft(nnx);
mag_nnxf=abs(nnx_f);
f=(0:nn-1)*fss/nn;
plot(f(1:nn/2),mag_nnxf(1:nn/2)*2/nn);
title('混合信号+噪声频谱');
%}

%% EMD算法实现
%{
imf=emd(d);
figure;
[a,b]=size(imf);
for i=1:a
    subplot(a,1,i);
    plot(imf(i,:));
end
figure;
for i=1:a
    imfi=fft(imf(i,:),N);
    mag=abs(imfi);
    f=(0:N-1)*fs/N;
    subplot(a,1,i);
    plot(f(1:N/2),mag(1:N/2)*2/N);
end
%}
%% VMD算法实现
%
% some sample parameters for VMD
alpha = 2000;        % moderate bandwidth constraint 数据保真平衡参数，影响模态估计的带宽。
tau = 0;            % noise-tolerance (no strict fidelity enforcement) 双上升时间间隔，当为0时，减弱噪声的影响
K = 2;              % 4 modes 模态分解的个数
DC = 0;             % no DC part imposed 
init = 1;           % initialize omegas uniformly 初始化所有中心频率
tol = 1e-7;         % 收敛条件

vmf=VMD(yn,alpha,tau,K,DC,init,tol);
figure;
[a,b]=size(vmf);
for i=1:a
    subplot(a,1,i);
    plot(vmf(i,:));
end
mag=zeros(a,N/2);
figure;
for i=1:a
    vmfi=fft(vmf(i,:),N);
    mag_v=abs(vmfi);
    mag(a,:)=mag_v(1:N/2);
    f=(0:N-1)*fs/N;
    subplot(a,1,i);
    plot(f(1:N/2),mag_v(1:N/2)*2/N);
end

[~,ff,Tf]=hhspectrum(mag(1,:));
[~,ff2,Tf2]=hhspectrum(mag(2,:));

figure;
subplot(211);
plot(Tf/fs,ff*fs);
title('瞬时频率');
subplot(212);
plot(Tf2/fs,ff2*fs);
%}
