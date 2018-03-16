clc;
clear all;

%% ���������ռ�������Χ
x=20;
y=20;
%% ���ø���Դλ��
center=[x/2,y/2];

%% ���ô��нڵ�λ�ã��������ռ�������ֲ�
num_detect=200;
rand_x=rand(num_detect,1)*x;
rand_y=rand(num_detect,1)*y;
detect_pos=[rand_x,rand_y];

%% ���ö����ʲ���
lost=zeros(num_detect,1);


%% ���ø���Դ�봫�н���ʼ����
P_center=20;
P_detect=0;    % ��λΪdBm

%% ���ø���Դ�ź�Ƶ��
a=[];
for i=1:num_detect
    distance=norm(detect_pos(i,:)-center);
    if distance>0.9 && distance<1.5
        a=[a;detect_pos(i,:)];
    end
end

%% ��˹ģ��
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
