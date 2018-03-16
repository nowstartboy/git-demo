clc;
clear all;

fs=50;
t=0:1/fs:1023/fs; %�������Ϊ0.02s,������Ƶ��Ϊ50Hz
N=length(t);
noise=randn(1,N);
I=cos(1.764*sin(0.7*pi*t+pi/6)+0.22*sin(2.24*pi*t+pi/3)+1.5*pi)+noise;  %�����I·�ź�
Q=sin(1.764*sin(0.7*pi*t+pi/6)+0.22*sin(2.24*pi*t+pi/3)+1.5*pi)+noise;  %�����Q·�ź�

%% ����ԭʼIQʱ��ͼ
figure(1)
subplot(211)
plot(t,I)
ylabel('I[n]')
subplot(212)
plot(t,Q)
ylabel('Q[n]')


%% 1��ʹ��CSD���
complex_signal=I+1j*Q;
complex_spectrum=fft(complex_signal);
fI=(0:N-1)*fs/N;  %Ƶ�׵��Ӧ��Ƶ��
figure(2)
%plot(fI(1:N/2),complex_spectrum(1:N/2)); %Ƶ��ǰ��κͺ����ǶԳƵģ�����ֻ��һ��
plot(fI(5:round(3000/fs)),abs(complex_spectrum(5:round(3000/fs))));   %ֻ��ǰ3Hz��Ƶ��
title('CSD Algorithm');

%% 2��ʹ��DACM���
Accum_deriva=zeros(1,N-1);
for i=2:N
    Accum_deriva(i-1)=(I(i)*(Q(i)-Q(i-1))-Q(i)*(I(i)-I(i-1)))/(I(i)^2+Q(i)^2);
end
phase_arc=zeros(1,N-1);    %DACM�������������ź�
for i=1:N-1
    phase_arc(i)=sum(Accum_deriva(1:i));
end
arc_spectrum=fft(phase_arc);
fI=(0:N-2)*fs/(N-1);  %Ƶ�׵��Ӧ��Ƶ��
figure(3)
%plot(fI(1:N/2),complex_spectrum(1:N/2)); %Ƶ��ǰ��κͺ����ǶԳƵģ�����ֻ��һ��
plot(fI(5:round(3000/fs)),abs(arc_spectrum(5:round(3000/fs))));   %ֻ��ǰ3Hz��Ƶ��
title('DACM Algorithm')

%% 3��ʹ�ò��������������Ⱥ�Ż��㷨
complex_signal=I+1j*Q;
x_range=[-5,5;0.15,0.6;0,2*pi];
Pg=Vital_particleSwarm(complex_signal,t,x_range);

% ������ռ�����
disp('������ȫ������λ��Ϊ��')
fprintf('��ǰ���ŵ�Ϊ��(%s,%s,%s)\n',Pg(1),Pg(2),Pg(3));

%{
c1=1.4962;
c2=1.4962;
w=0.7298;
maxDt=100;    %����������
D=3;           %δ֪������
N=40;          %����Ⱥ�����
eps=10^(-7);   %��������

% ��ʼ����Ⱥ�ĸ���
x=randn(N,D);  %��ʼ����Ⱥλ��
v=randn(N,D);  %��ʼ����Ⱥ�ٶ�

% �ȼ��������ʼ���ӵ���Ӧ�ȣ�ȷ�������ʼ����λ��y��ȫ������λ��Pg
p=zeros(1,N);
y=zeros(N,D);
for i=1:N
    p(i)=sphere(complex_signal,t,x(i,:));  %������Ӧ�ȣ��Ż�����Ϊsphere
    y(i,:)=x(i,:);          %��������λ��Ϊt=0ʱ���λ��
end
[max_value,max_pos]=max(p);
Pg=x(max_pos,:);            %ȫ������λ��
Pg_best=max_value;       %ȫ������ֵ

% ѵ���������ҵ�����λ�ã�ֱ�����㾫��Ҫ����ߵ����������
for iter=1:maxDt
    for i=1:N
        v(i,:)=w*v(i,:)+c1*rand(1)*(y(i,:)-x(i,:))+c2*rand(1)*(Pg-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        px=sphere(complex_signal,t,x(i,:));
        if px>p(i)
            p(i)=px;    %������Ӧ��
            y(i,:)=x(i,:);  %���¸������λ��
        end
        if p(i)>Pg_best
            Pg=y(i,:);
            Pg_best=p(i);
        end
    end
    if mod(iter,10)==0
        disp('**************************************');
        fprintf('��ǰ��(%d)�ε������ŵ�Ϊ��(%s,%s,%s)\n',iter,Pg(1),Pg(2),Pg(3));
    end
end

% ������ռ�����
disp('������ȫ������λ��Ϊ��')
fprintf('��ǰ���ŵ�Ϊ��(%s,%s,%s)\n',Pg(1),Pg(2),Pg(3));
%}

%% 3.1 �ڶ���ʹ������Ⱥ�㷨������������������г��
disp('****************************************************************')
disp('***************���������������������κ���г��***********************')

Sigma_t=exp(-1j*Pg(1)*sin(2*pi*Pg(2)*t+Pg(3)));
complex_signal2=complex_signal.*Sigma_t;
x_range1=[-5,5;2*Pg(2)-0.034,2*Pg(2)+0.034;0,2*pi]; %����г����ȡֵ��Χ
Pg1=Vital_particleSwarm(complex_signal2,t,x_range1);

% ������ռ�����
disp('������ȫ������λ��Ϊ��')
fprintf('��ǰ���ŵ�Ϊ��(%s,%s,%s)\n',Pg1(1),Pg1(2),Pg1(3));

%{
c1=1.4962;
c2=1.4962;
w=0.7298;
maxDt=100;    %����������
D=3;           %δ֪������
N=40;          %����Ⱥ�����
eps=10^(-7);   %��������

% ��ʼ����Ⱥ�ĸ���
x=randn(N,D);  %��ʼ����Ⱥλ��
v=randn(N,D);  %��ʼ����Ⱥ�ٶ�

% �ȼ��������ʼ���ӵ���Ӧ�ȣ�ȷ�������ʼ����λ��y��ȫ������λ��Pg
p=zeros(1,N);
y=zeros(N,D);
for i=1:N
    p(i)=sphere(complex_signal2,t,x(i,:));  %������Ӧ�ȣ��Ż�����Ϊsphere
    y(i,:)=x(i,:);          %��������λ��Ϊt=0ʱ���λ��
end
[max_value,max_pos]=max(p);
Pg1=x(max_pos,:);            %ȫ������λ��
Pg_best=max_value;       %ȫ������ֵ

% ѵ���������ҵ�����λ�ã�ֱ�����㾫��Ҫ����ߵ����������
for iter=1:maxDt
    for i=1:N
        v(i,:)=w*v(i,:)+c1*rand(1)*(y(i,:)-x(i,:))+c2*rand(1)*(Pg1-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        px=sphere(complex_signal2,t,x(i,:));
        if px>p(i)
            p(i)=px;    %������Ӧ��
            y(i,:)=x(i,:);  %���¸������λ��
        end
        if p(i)>Pg_best
            Pg1=y(i,:);
            Pg_best=p(i);
        end
    end
    if mod(iter,10)==0
        disp('**************************************');
        fprintf('��ǰ��(%d)�ε������ŵ�Ϊ��(%s,%s,%s)\n',iter,Pg1(1),Pg1(2),Pg1(3));
    end
end

% ������ռ�����
disp('������ȫ������λ��Ϊ��')
fprintf('��ǰ���ŵ�Ϊ��(%s,%s,%s)\n',Pg1(1),Pg1(2),Pg1(3));
%}

%% 3.2 ��ȡ����������������������������г��
disp('****************************************************************')
disp('***************����������������������г������***********************')

Sigma_t=exp(-1j*Pg1(1)*sin(2*pi*Pg1(2)*t+Pg1(3)));
complex_signal3=complex_signal2.*Sigma_t;
x_range2=[-5,5;3*Pg(2)-0.051,3*Pg(2)+0.051;0,2*pi];  %����г��ȡֵ��Χ
Pg2=Vital_particleSwarm(complex_signal3,t,x_range2);

% ������ռ�����
disp('������ȫ������λ��Ϊ��')
fprintf('��ǰ���ŵ�Ϊ��(%s,%s,%s)\n',Pg2(1),Pg2(2),Pg2(3));

%{
c1=1.4962;
c2=1.4962;
w=0.7298;
maxDt=100;    %����������
D=3;           %δ֪������
N=40;          %����Ⱥ�����
eps=10^(-7);   %��������

% ��ʼ����Ⱥ�ĸ���
x=randn(N,D);  %��ʼ����Ⱥλ��
v=randn(N,D);  %��ʼ����Ⱥ�ٶ�

% �ȼ��������ʼ���ӵ���Ӧ�ȣ�ȷ�������ʼ����λ��y��ȫ������λ��Pg
p=zeros(1,N);
y=zeros(N,D);
for i=1:N
    p(i)=sphere(complex_signal3,t,x(i,:));  %������Ӧ�ȣ��Ż�����Ϊsphere
    y(i,:)=x(i,:);          %��������λ��Ϊt=0ʱ���λ��
end
[max_value,max_pos]=max(p);
Pg2=x(max_pos,:);            %ȫ������λ��
Pg_best=max_value;       %ȫ������ֵ

% ѵ���������ҵ�����λ�ã�ֱ�����㾫��Ҫ����ߵ����������
for iter=1:maxDt
    for i=1:N
        v(i,:)=w*v(i,:)+c1*rand(1)*(y(i,:)-x(i,:))+c2*rand(1)*(Pg2-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        px=sphere(complex_signal3,t,x(i,:));
        if px>p(i)
            p(i)=px;    %������Ӧ��
            y(i,:)=x(i,:);  %���¸������λ��
        end
        if p(i)>Pg_best
            Pg2=y(i,:);
            Pg_best=p(i);
        end
    end
    if mod(iter,10)==0
        disp('**************************************');
        fprintf('��ǰ��(%d)�ε������ŵ�Ϊ��(%s,%s,%s)\n',iter,Pg2(1),Pg2(2),Pg2(3));
    end
end

% ������ռ�����
disp('������ȫ������λ��Ϊ��')
fprintf('��ǰ���ŵ�Ϊ��(%s,%s,%s)\n',Pg2(1),Pg2(2),Pg2(3));
%}

%% 3.3 ����������������г��Ȼ������Ѱ����������
disp('****************************************************************')
disp('***************��������������г��������Ѱ����������***********************')
    
Sigma_t=exp(-1j*Pg2(1)*sin(2*pi*Pg2(2)*t+Pg2(3)));
complex_signal4=complex_signal3.*Sigma_t;
x_range3=[-5,5;0.8,2;0,2*pi];       %����ȡֵ��Χ
Pg3=Vital_particleSwarm(complex_signal4,t,x_range3);

% ������ռ�����
disp('������ȫ������λ��Ϊ��')
fprintf('��ǰ���ŵ�Ϊ��(%s,%s,%s)\n',Pg3(1),Pg3(2),Pg3(3));


%{
c1=1.4962;
c2=1.4962;
w=0.7298;
maxDt=100;    %����������
D=3;           %δ֪������
N=40;          %����Ⱥ�����
eps=10^(-7);   %��������

% ��ʼ����Ⱥ�ĸ���
x=randn(N,D);  %��ʼ����Ⱥλ��
v=randn(N,D);  %��ʼ����Ⱥ�ٶ�

% �ȼ��������ʼ���ӵ���Ӧ�ȣ�ȷ�������ʼ����λ��y��ȫ������λ��Pg
p=zeros(1,N);
y=zeros(N,D);
for i=1:N
    p(i)=sphere(complex_signal4,t,x(i,:));  %������Ӧ�ȣ��Ż�����Ϊsphere
    y(i,:)=x(i,:);          %��������λ��Ϊt=0ʱ���λ��
end
[max_value,max_pos]=max(p);
Pg3=x(max_pos,:);            %ȫ������λ��
Pg_best=max_value;       %ȫ������ֵ

% ѵ���������ҵ�����λ�ã�ֱ�����㾫��Ҫ����ߵ����������
for iter=1:maxDt
    for i=1:N
        v(i,:)=w*v(i,:)+c1*rand(1)*(y(i,:)-x(i,:))+c2*rand(1)*(Pg3-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        px=sphere(complex_signal4,t,x(i,:));
        if px>p(i)
            p(i)=px;    %������Ӧ��
            y(i,:)=x(i,:);  %���¸������λ��
        end
        if p(i)>Pg_best
            Pg3=y(i,:);
            Pg_best=p(i);
        end
    end
    if mod(iter,10)==0
        disp('**************************************');
        fprintf('��ǰ��(%d)�ε������ŵ�Ϊ��(%s,%s,%s)\n',iter,Pg3(1),Pg3(2),Pg3(3));
    end
end

% ������ռ�����
disp('������ȫ������λ��Ϊ��')
fprintf('��ǰ���ŵ�Ϊ��(%s,%s,%s)\n',Pg3(1),Pg3(2),Pg3(3));
%}

%% �������������˲���Ĳ���
figure(4);
complex_spectrum=abs(fft(complex_signal));
complex_spectrum=complex_spectrum/max(complex_spectrum);
complex_spectrum2=abs(fft(complex_signal2));
complex_spectrum2=2*complex_spectrum2/max(complex_spectrum2);
complex_spectrum3=abs(fft(complex_signal3));
complex_spectrum3=complex_spectrum3/max(complex_spectrum3);
fI=(0:N-1)*fs/N;  %Ƶ�׵��Ӧ��Ƶ��

%plot(fI(1:N/2),complex_spectrum(1:N/2)); %Ƶ��ǰ��κͺ����ǶԳƵģ�����ֻ��һ��
t1=fI(1:round(3000/fs));
plot(t1,complex_spectrum(1:round(3000/fs)),'b',t1,complex_spectrum2(1:round(3000/fs)) ,'r',t1,complex_spectrum3(1:round(3000/fs)),'g');   %ֻ��ǰ3Hz��Ƶ��
legend('ԭʼ���ź�Ƶ��','�����˺���Ƶ��','��������������г��');
title('����˳�г���������');
