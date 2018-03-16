clc;
clear all;

%% 设置整个空间的面积范围
x=20;
y=20;
%% 设置干扰源位置
center=[x/2,y/2];

%% 设置传感节点位置，在整个空间内随机分布
num_detect=200;
rand_x=rand(num_detect,1)*x;
rand_y=rand(num_detect,1)*y;
detect_pos=[rand_x,rand_y];

%% 设置丢包率参数
lost=zeros(num_detect,1);


%% 设置干扰源与传感结点初始功率
P_center=20;
P_detect=0;    % 单位为dBm

%% 设置干扰源信号频率
a=[];
for i=1:num_detect
    distance=norm(detect_pos(i,:)-center);
    if distance>0.9 && distance<1.5
        a=[a;detect_pos(i,:)];
    end
end

%% 高斯模型
GMModel = fitgmdist(a,1); %fit GMM distribution 
GMModel.mu
size_a=size(a);
y = zeros(size_a(1),1);  
h = gscatter(a(:,1),a(:,2),y); 
figure;
h = gscatter(a(:,1),a(:,2),y); 
hold on;  
ezcontour (@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}));  
title('{\bf Fitted Gaussian Mixture Contours}');  
legend(h,'Model 0','Model1')  
hold off;  
