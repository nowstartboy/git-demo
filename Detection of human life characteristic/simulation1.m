% clc;
% clear all;

%% ģ�������ͺ����ź�
%����Ƶ��1.4HZ������0.05������Ƶ��Ϊ0.4HZ������Ϊ1
A1=0.1;
f1=1.4;
A2=1;
f2=0.4;
fs=20; %����Ƶ��
dt=1/fs; %����ʱ����
n=1024;  %��������
t=dt:dt:n/fs;
s1=A1*sin(2*pi*f1*t);
s2=A2*sin(2*pi*f2*t);
subplot(211);
plot(s1);
y_s1=hilbert(s1);
hold on;
plot(abs(y_s1),'r');
title('����ʱ��ͼ');
subplot(212);
plot(s2);
y_s2=hilbert(s2);
hold on;
plot(abs(y_s2),'r');
title('����ʱ��ͼ');

%����ź�
figure;
d=s1+s2;
N=n;         %N-fft���õ����ݸ���
d_f=fft(d,N);  
subplot(211);
plot(d);
title('����ź�ͼ');
subplot(212);
mag=abs(d_f);
f=(0:N-1)*fs/N;
plot(f(1:N/2),mag(1:N/2)*2/N);
title('����ź�Ƶ��');

%������������ź�
figure;
fn=5;
fi=randn(1,n);
sn=0.7*cos(2*pi*fn*t); %ϵͳ���Զ�������
d=s1+s2;
dn=d;  %û���յ����ŵ��ź�

%���ּӰ������ķ���
%1
% d=awgn(d,10);
% dn1=d; %���������������ź�

%2
%noise=wgn(1,n,-5);
%d=d+noise;

%3
%��ָ������Ȱ�����
SNR=-3;
noise=randn(1,n);
signal_power=(1/n)*sum(d.*d);
noise_variance=signal_power/(10^(SNR/10));
noise=sqrt(noise_variance)/std(noise)*noise;
d=d+noise;
dn1=d;
%end

d=d+sn;   %�������͸����ź�
dd=dn+sn; %û��������ֻ�и����ź�
N=n;         %N-fft���õ����ݸ���
d_f=fft(d,N);  
subplot(211);
plot(d);
title('����ź�+����ͼ');
subplot(212);
mag=abs(d_f);
f=(0:N-1)*fs/N;
plot(f(1:N/2),mag(1:N/2)*2/N);
title('����ź�+����Ƶ��');

%% LMS�����˲�
%{
% ʹ����ͨLMS�㷨�����˲�
M1=80;
xn=d';
en = zeros(n,1);             % �������,en(k)��ʾ��k�ε���ʱԤ�������ʵ����������
W  = zeros(M1,1);             % ÿһ�д���һ����Ȩ����,ÿһ�д���-�ε���,��ʼΪ0
yn=zeros(size(xn));
%yn(1:M1-1)=xn(1:M1-1);

rho_max=max(eig(xn*xn'));
mu=rand()*(1/rho_max);
%mu=0.001;

%FIR+LMS�˲���
% [b,a]=butter(8,0.25);
% dn=filter(b,a,dd);


% LMS��������
for k = M1:n                  % ��k�ε���
    x = xn(k:-1:k-M1+1);        % �˲���M����ͷ������
    yn(k) = W.' * x;        % �˲��������
    en(k) = dn(k) - yn(k) ;        % ��k�ε��������
    
    % �˲���Ȩֵ����ĵ���ʽ
    W = W + 2*mu*en(k)*x;
    % ��һ��LMS
    %W = W + (mu/norm(x))*en(k)*x;
end
%}

%% RLS�㷨�����˲�
%
M1=30;
xn=dn';
en = zeros(n,1);             % �������,en(k)��ʾ��k�ε���ʱԤ�������ʵ����������
W  = zeros(M1,1);             % ÿһ�д���һ����Ȩ����,ÿһ�д���-�ε���,��ʼΪ0
T=eye(M1,M1)*10;
yn=zeros(size(xn));
%yn(1:M1-1)=xn(1:M1-1);

rho_max=max(eig(xn*xn'));
mu=rand()*(1/rho_max);
%mu=0.001;
lambda=0.98; %��������

%FIR+LMS�˲���
% [b,a]=butter(8,0.25);
% dn=filter(b,a,dd);

for k = M1+1:n                  % ��k�ε���
    x = xn(k-1:-1:k-M1);        % �˲���M����ͷ������
    K=(T*x)/(lambda+x'*T*x);
    en(k)=dn(k)-W'*x;       % ��k�ε��������
    W=W+K*en(k);
    yn(k) = W.' * x;        % �˲��������      
    T=(T-K*x'*T)/lambda;
end
dlmwrite('W.txt',W);
%}

figure;
% plot(t,yn,'b',t,dn,'g',t,en,'r');
% legend('�˲������','Ԥ�����','���');
subplot(311)
plot(t,yn,'b');
title('�˲������');
subplot(312)
plot(t,dn,'g');
title('Ԥ�����');
subplot(313);
plot(t,en,'r');
title('���');

figure;
yn_f=fft(yn,N);
dn_f=fft(dn,N);
subplot(211)
mag_yn=abs(yn_f);
plot(f(1:N/2),mag_yn(1:N/2));
title('�˲������Ƶ��');
subplot(212)
mag_dn=abs(dn_f);
plot(f(1:N/2),mag_dn(1:N/2));
title('Ԥ�����Ƶ��');



%% ��̬MTI �˲����˳������Ӳ���
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
title('�����ź�ʱ��Ƶ��ͼ');
subplot(212);
xx_f=fft(xx);
mag_xxf=abs(xx_f);
f=(0:nn-1)*fss/nn;
plot(f(1:nn/2),mag_xxf(1:nn/2)*2/nn);
title('����ź�+����Ƶ��');

figure;
subplot(211);
plot(nnx)
title('�����ź�ʱ��Ƶ��ͼ');
subplot(212);
nnx_f=fft(nnx);
mag_nnxf=abs(nnx_f);
f=(0:nn-1)*fss/nn;
plot(f(1:nn/2),mag_nnxf(1:nn/2)*2/nn);
title('����ź�+����Ƶ��');
%}

%% EMD�㷨ʵ��
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
%% VMD�㷨ʵ��
%
% some sample parameters for VMD
alpha = 2000;        % moderate bandwidth constraint ���ݱ���ƽ�������Ӱ��ģ̬���ƵĴ���
tau = 0;            % noise-tolerance (no strict fidelity enforcement) ˫����ʱ��������Ϊ0ʱ������������Ӱ��
K = 2;              % 4 modes ģ̬�ֽ�ĸ���
DC = 0;             % no DC part imposed 
init = 1;           % initialize omegas uniformly ��ʼ����������Ƶ��
tol = 1e-7;         % ��������

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
title('˲ʱƵ��');
subplot(212);
plot(Tf2/fs,ff2*fs);
%}
