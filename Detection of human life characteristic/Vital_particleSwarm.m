function Pg = Vital_particleSwarm(complex_signal,t,x_range)
%% ��������Ⱥ�㷨�������ƥ��������������������ȣ�Ƶ�ʣ���λ��
%% ������
%    complex_signal��   %ԭʼ�������ź�
%    t��������
%    x_range���޶����������ı仯��Χ

%% ����ֵ����ƥ��������������������ȣ�Ƶ�ʣ���λ��
%% �������壺
c1=1.4962;
c2=1.4962;    %�������Ȳ���������ˡ�
w=0.7298;
maxDt=300;    %����������
D=3;           %δ֪������
N=40;          %����Ⱥ�����
eps=10^(-7);   %��������

% ��ʼ����Ⱥ�ĸ���
x=zeros(N,D);  %��ʼ����Ⱥλ��
%��ָ�������ڳ�ʼ������
for i=1:D
    x(:,i)=x_range(i,1)+(x_range(i,2)-x_range(i,1))*rand(N,1);
end
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
fprintf('��ǰ��(0)�ε������ŵ�Ϊ��(%s,%s,%s)\n',Pg(1),Pg(2),Pg(3));

% ѵ���������ҵ�����λ�ã�ֱ�����㾫��Ҫ����ߵ����������
for iter=1:maxDt
    for i=1:N
        v(i,:)=w*v(i,:)+c1*rand(1)*(y(i,:)-x(i,:))+c2*rand(1)*(Pg-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        %���Ʋ�����ָ������
        for j=1:D
            if x(i,j)<x_range(j,1)
                %x(i,j)=x_range(j,1);            %�߽�̶�
                x(i,j)=x_range(j,1)+x_range(j,2)-x_range(j,1); %ѭ������
            end
            if x(i,j)>x_range(j,2)
                %x(i,j)=x_range(j,2);
                x(i,j)=x_range(j,2)-x_range(j,2)+x_range(j,1);
            end
        end
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
    if mod(iter,20)==0
        disp('**************************************');
        fprintf('��ǰ��(%d)�ε������ŵ�Ϊ��(%s,%s,%s)\n',iter,Pg(1),Pg(2),Pg(3));
    end
end

% ������ռ�����
disp('������ȫ������λ��Ϊ��')
fprintf('��ǰ���ŵ�Ϊ��(%s,%s,%s)\n',Pg(1),Pg(2),Pg(3));
end

