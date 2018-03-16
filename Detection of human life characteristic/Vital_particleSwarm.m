function Pg = Vital_particleSwarm(complex_signal,t,x_range)
%% 利用粒子群算法来算出最匹配的生命体征参数（幅度，频率，相位）
%% 参数：
%    complex_signal：   %原始的体征信号
%    t：采样点
%    x_range：限定搜索变量的变化范围

%% 返回值：最匹配的生命体征参数（幅度，频率，相位）
%% 程序主体：
c1=1.4962;
c2=1.4962;    %参数就先不设成输入了。
w=0.7298;
maxDt=300;    %最大迭代次数
D=3;           %未知量个数
N=40;          %粒子群体个数
eps=10^(-7);   %迭代精度

% 初始化种群的个体
x=zeros(N,D);  %初始化种群位置
%在指定区间内初始化参数
for i=1:D
    x(:,i)=x_range(i,1)+(x_range(i,2)-x_range(i,1))*rand(N,1);
end
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
fprintf('当前第(0)次迭代最优点为：(%s,%s,%s)\n',Pg(1),Pg(2),Pg(3));

% 训练迭代，找到最优位置，直到满足精度要求或者迭代次数完成
for iter=1:maxDt
    for i=1:N
        v(i,:)=w*v(i,:)+c1*rand(1)*(y(i,:)-x(i,:))+c2*rand(1)*(Pg-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        %限制参数在指定区间
        for j=1:D
            if x(i,j)<x_range(j,1)
                %x(i,j)=x_range(j,1);            %边界固定
                x(i,j)=x_range(j,1)+x_range(j,2)-x_range(j,1); %循环遍历
            end
            if x(i,j)>x_range(j,2)
                %x(i,j)=x_range(j,2);
                x(i,j)=x_range(j,2)-x_range(j,2)+x_range(j,1);
            end
        end
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
    if mod(iter,20)==0
        disp('**************************************');
        fprintf('当前第(%d)次迭代最优点为：(%s,%s,%s)\n',iter,Pg(1),Pg(2),Pg(3));
    end
end

% 输出最终计算结果
disp('函数的全局最优位置为：')
fprintf('当前最优点为：(%s,%s,%s)\n',Pg(1),Pg(2),Pg(3));
end

