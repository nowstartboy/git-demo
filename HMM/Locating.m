clc;
clear all;

%% 设置整个空间的面积范围
x1=20;
y1=20;
%% 设置干扰源位置
center1=[x1*2/10,y1*3/10]; %第一个干扰源
center2=[x1*8/10,y1*7/10];   %第二个干扰源

%% 设置传感节点位置，在整个空间内随机分布
num_detect=300;
rand_x=rand(num_detect,1)*x1;
rand_y=rand(num_detect,1)*y1;
detect_pos=[rand_x,rand_y];
one_pos=ones(num_detect,1);
detect=[one_pos,rand_x,rand_y,rand_x.^2,rand_y.^2,rand_x.*rand_y,rand_x.^3,rand_y.^3,rand_x.^2.*rand_y,rand_y.^2.*rand_x];

%% 设置丢包率参数
lost=zeros(num_detect,1);

%% 设置干扰源与传感结点初始功率dBm
P_center1=20;
P_center2=15;

%设置阈值，当干扰功率大于某一定程度的时候，就不算在计算范围内
P_thera=P_center1/2;

%% 设置干扰源信号频率
a=[];
distance=zeros(num_detect,1);
for i=1:num_detect
    distance1=norm(detect_pos(i,:)-center1);
    distance(i)=distance1;
    distance2=norm(detect_pos(i,:)-center2);
    lost(i)=P_center1-15*log10(1+distance1);
    %lost(i)=lost(i)+P_center2-15*log10(1+distance2);
    if lost(i)<=P_thera && lost(i)>=P_thera/4 %在阈值以下的结点，都默认为收不到信息
        a=[a;i];
    end
end


ya=[detect_pos(a,:),distance(a),lost(a)];
yaa=sortrows(ya,4);

scatter(detect_pos(a,1),detect_pos(a,2),'r');
hold on;

[max_t,max_no]=sort(lost(a));
stop=length(max_t);
stop_max=max_t(end);
for i=length(max_t):-1:1
    if abs(max_t(i)-stop_max)>1.2
        stop=i;
        break;
    end
end
 
max_pos=yaa(stop:end,1:2);
scatter(max_pos(:,1),max_pos(:,2),'g');
figure;

x_pos=[min(max_pos(:,1)),max(max_pos(:,1))];
y_pos=[min(max_pos(:,2)),max(max_pos(:,2))];


%% 回归拟合
[b,bint,r,rint,stats]=regress(lost(a),detect(a,:));

%% 画出拟合曲面
dt=0.01;
t1=x_pos(1)+dt:dt:x_pos(2);
t2=y_pos(1)+dt:dt:y_pos(2);

x_total=int16((x_pos(2)-x_pos(1))/dt)-1;
y_total=int16((y_pos(2)-y_pos(1))/dt)-1;
z=zeros(x_total,y_total);
for i=1:x_total
    for j=1:y_total
        z(i,j)=b(1)+b(2)*t1(i)+b(3)*t2(j)+b(4)*t1(i)^2+b(5)*t2(j)^2+b(6)*t1(i)*t2(j)+b(7)*t1(i).^3+b(8)*t2(j).^3+b(9)*t1(i).^2*t2(j)+b(10)*t1(i)*t2(j).^2;
    end
end
[ux,uy]=find(z==max(max(z)));
fprintf ('\nthe exact result is (%f,%f)\n ',double(ux*dt+x_pos(1)),double(uy*dt+y_pos(1)));
mesh(z);

%% 求解拟合方程峰值，共轭梯度法或牛顿法
syms x;
syms y;
f=b(1)+b(2)*x+b(3)*y+b(4)*x^2+b(5)*y^2+b(6)*x*y+b(7)*x^3+b(8)*y^3+b(9)*x^2*y+b(10)*x*y^2;
%fprintf ('\nthe exact result is (%f,%f)\n ',double(-2*b(4)/b(2)),double(-2*b(5)/b(3)));
x0=[10,10];

%% 共轭梯度法
%{
n0=40;
t0=clock; %时间戳，记录整段代码运行时间
error=0.001;  %阈值
v=[x,y];
f_d=jacobian(f,v);
d=-subs(f_d,v,x0);
d_A=norm(double(d));
xk=x0;
f_value=subs(f,v,x0);
syms lambda;
n=1; %迭代次数
while n<=n0
    fprintf ('\nthe NO.%d iter..... \n ',n);
    n=n+1;
    d_xk=d;
    fprintf ('the d is (%f,%f) \n',double(d_xk(1)),double(d_xk(2)));
    xk=xk+lambda*d;
    f_lambda=subs(f,{x,y},{xk(1),xk(2)});
    df=diff(f_lambda);
    d_lambda=double(solve(df));
    d_lambda=d_lambda(~logical(imag(d_lambda))); %只保留实数解
    f_lambda_num=double(subs(f_lambda,d_lambda));
    [a,b]=min(f_lambda_num);
    d_lambda=d_lambda(b);
    fprintf ('the lambda is %f \n',double(d_lambda));
    xk=double(subs(xk,lambda,d_lambda));
    d_xk_plus=double(subs(f_d,v,xk));
    fprintf ('the xk_plus is (%f,%f) \n',double(d_xk_plus(1)),double(d_xk_plus(2)));
    d_A=norm(d_xk_plus);
    f_value=[f_value,double(subs(f,v,xk))];
    if d_A<error
        fprintf ('the result is (%f,%f) \n',double(d_xk_plus(1)),double(d_xk_plus(2)));
        fprintf ('the end \n');
        break;
    else
        beta=norm(double(d_xk_plus))^2/norm(double(d_xk))^2;
        fprintf ('the beta is %f \n',double(beta));
        d=-d_xk_plus+beta*d_xk;
    end
    fprintf ('the xk is (%f,%f) \n',double(xk(1)),double(xk(2)));
    fprintf ('the d_A is %f \n',double(d_A));
end
elapsed_time=etime(clock,t0);
fprintf ('the d_A is %f \n',double(d_A));
fprintf ('the xk is (%f,%f) \n',double(xk(1)),double(xk(2)));
fprintf ('the num of iters is %d \n',n);
fprintf ('the cost of time is %f \n',elapsed_time);

%}

%% 牛顿法
%{
t0=clock;
error=10^(-4);  %阈值
v=[x,y];
f_value=subs(f,v,x0);
f_d=jacobian(f,v);
f_d2=jacobian(f_d,v);
d_1=subs(f_d,v,x0);
d_2=subs(f_d2,v,x0);
d_A=norm(double(d_1));
xk=x0';
n=0; %迭代次数
while d_A>error
    n=n+1;
    fprintf ('the f_d1 is (%f,%f) \n',double(d_1(1)),double(d_1(2)));
    fprintf ('the f_d2 is (%f,%f,%f,%f) \n',double(d_2(1,1)),double(d_2(1,2)),double(d_2(2,1)),double(d_2(2,2)));
    en=d_2\d_1';
    fprintf ('the inv(d_2)*d_1 is (%f,%f) \n',double(en(1)),double(en(2)));
    xk=xk-en;
    fprintf ('the xk is (%f,%f) \n',double(xk(1)),double(xk(2)));
    d_1=double(subs(f_d,v,xk'));
    fprintf ('the d_1 is (%f,%f) \n',double(d_1(1)),double(d_1(2)));
    d_2=double(subs(f_d2,v,xk'));
    d_A=norm(d_1);
    f_value=[f_value,double(subs(f,v,xk'))];
    fprintf ('the d_A is %f \n',double(d_A));
    fprintf ('the xk_plus is (%f,%f) \n',double(xk(1)),double(xk(2)));
    fprintf ('the xk is (%f,%f) \n',double(xk(1)),double(xk(2)));
end
elapsed_time=etime(clock,t0);
fprintf ('the num of iters is %d \n',n);
fprintf ('the cost of time is %f \n',elapsed_time);
%}

%% 高斯模型
len_a=length(yaa);
a1=yaa(fix(len_a/4):end,[1,2,4]);
%a1=[detect_pos(a,:),lost(a)];
GMModel = fitgmdist(a1,2); %fit GMM distribution 
GMModel.mu
fprintf ('\nthe GMM exact result is (%f,%f)\n ',double(GMModel.mu(1)),double(GMModel.mu(2)));


size_a=size(a1);
y = zeros(size_a(1),1);  
h = gscatter(a1(:,1),a1(:,2),y);
figure;
hold on;
%ezcontour (@(x1,x2)pdf(GMModel,[x1 x2]),[0,20],[0,20]);  
scatter3(a1(:,1),a1(:,2),a1(:,3));
hold off;
% 
% hold on;  
% get(gca,{'XLim','YLim','ZLim'});
% ezcontour (@(x1,x2,x3)pdf(GMModel,[x1 x2,x3]),[0,20],[0,20],[0,20]);  
% title('{\bf Fitted Gaussian Mixture Contours}');  
% legend(h,'Model 0','Model1')  
% hold off;  



