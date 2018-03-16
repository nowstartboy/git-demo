clc;
clear all;
%% 设置区域范围
x_=9;
y_=9;

%% 设置传感点均匀分布
num_detect_even=400;
% det_pos=zeros(num_detect_even,2);
% d=sqrt(num_detect_even/100);
% for i=0:d*x_+d-1
%     for j=0:d*y_+d-1
%         det_pos(10*d*i+j+1,:)=[i/d,j/d];
%     end
% end

%设置传感节点位置，在整个空间内随机分布
num_detect=400;
rand_x=rand(num_detect,1)*(x_*2);
rand_y=rand(num_detect,1)*(y_*2);
det_pos=[rand_x,rand_y];

%% 干扰源位置随机分布,设置多个干扰源，对最终预测误差进行MSE分析
num_disturb=20;
rand_x=rand(num_disturb,1)*x_;
rand_y=rand(num_disturb,1)*y_;
disturb_pos=[rand_x,rand_y];

%% 设置干扰源与传感结点初始功率
P_center=20;
P_detect=0;    % 单位为dBm

%% 信号传输模型，一般来说信号衰减与logD成正线性关系，所以可以直接用D的log10结果进行直接线性分析
%f=1e8;
%Ls=32.44+(20*log10(f))+(20*log10(D));

%% 进行仿真
disturb_pos_get=zeros(num_disturb,2); %对每个干扰源进行预测的结果
for i=1:num_disturb
    a=[];
    for j=1:num_detect_even
        if det_pos(j,1)>disturb_pos(i,1)-5 && det_pos(j,1)<disturb_pos(i,1)+5 && det_pos(j,2)>disturb_pos(i,2)-5 && det_pos(j,2)<disturb_pos(i,2)+5
            distance=norm(det_pos(j,:)-disturb_pos(i,:));
            if distance>1 && distance<2
                a=[a;det_pos(j,:)];
            end
        end
    end
    GMModel = fitgmdist(a,1); %fit GMM distribution 
    disturb_pos_get(i,:)=GMModel.mu;
end

%画出所有点
size_det=size(det_pos);
y_det=zeros(size_det(1),1);
h0=gscatter(det_pos(:,1),det_pos(:,2),y_det);
figure;

size_a=size(a);
y = zeros(size_a(1),1);  
h1 = gscatter(a(:,1),a(:,2),y);  
figure;

h = gscatter(a(:,1),a(:,2),y);  
hold on;  
get(gca,{'XLim','YLim'});
ezcontour (@(x1,x2)pdf(GMModel,[x1 x2]),[GMModel.mu(1)-3,GMModel.mu(1)+3],[GMModel.mu(2)-3,GMModel.mu(2)+3]);  
title('{\bf Fitted Gaussian Mixture Contours}');  
legend(h,'Model 0','Model1')  
hold off;  

figure;
error_pos=disturb_pos_get-disturb_pos;
error=sqrt((error_pos(:,1).^2+error_pos(:,2).^2));
plot(error);
title('predict-error')
xlabel('Number')
ylabel('error')
    

