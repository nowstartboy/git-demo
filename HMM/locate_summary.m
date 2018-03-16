clc;
clear all;

%% ���������ռ�������Χ
x1=10;
y1=10;
%% ���ø���Դλ��
center1=[x1*1/10,y1*1/10]; %��һ������Դ
center2=[x1*3/10,y1*7/10];   %�ڶ�������Դ

num_detect1=[20,50,100,200,300];
sum_locate=30;
error_no=zeros(2,sum_locate);
error_detect=zeros(1,num_detect1(3));

for ii=1:sum_locate

%% ���ô��нڵ�λ�ã��������ռ�������ֲ�
num_detect=100;
rand_x=rand(num_detect,1)*x1;
rand_y=rand(num_detect,1)*y1;
detect_pos=[rand_x,rand_y];
one_pos=ones(num_detect,1);
%detect=[one_pos,rand_x,rand_y,rand_x.^2,rand_y.^2,rand_x.*rand_y,rand_x.^3,rand_y.^3,rand_x.^2.*rand_y,rand_y.^2.*rand_x];
detect=get_multiFeature(detect_pos,5);

%% ���ö����ʲ���
lost=zeros(num_detect,1);

%% ���ø���Դ�봫�н���ʼ����dBm
P_center1=20;
P_center2=15;

%������ֵ�������Ź��ʴ���ĳһ���̶ȵ�ʱ�򣬾Ͳ����ڼ��㷶Χ��
P_thera=P_center1/2;

%% ���ø���Դ�ź�Ƶ��
a=[];
distance=zeros(num_detect,1);
for i=1:num_detect
    distance1=norm(detect_pos(i,:)-center1);
    distance(i)=distance1;
    distance2=norm(detect_pos(i,:)-center2);
    lost(i)=P_center1-15*log10(1+distance1);
    %lost(i)=lost(i)+P_center2-15*log10(1+distance2);
    %lost(i)=lost(i)+randn(1,1);
    if lost(i)<=P_thera && lost(i)>=P_thera/4 %����ֵ���µĽ�㣬��Ĭ��Ϊ�ղ�����Ϣ
        a=[a;i];
    end
end


ya=[detect_pos(a,:),distance(a),lost(a)];
yaa=sortrows(ya,4);



[max_t,max_no]=sort(lost(a));
stop=length(max_t);
stop_max=max_t(end);
for i=length(max_t):-1:1
    if abs(max_t(i)-stop_max)>1
        stop=i;
        break;
    end
end
%stop=1;
 
max_pos=yaa(stop:end,:);


x_pos=[min(max_pos(:,1)),max(max_pos(:,1))];
y_pos=[min(max_pos(:,2)),max(max_pos(:,2))];

A=[2*max_pos(:,1),2*max_pos(:,2),ones(size(max_pos,1),1)];
b1=max_pos(:,1).^2+max_pos(:,2).^2;

result1=(A'*A)\A'*b1;
fprintf ('\nthe exact result1 is (%f,%f)\n ',double(result1(1)),double(result1(2)));
error1=sqrt((center1(1)-double(result1(1)))^2+(center1(2)-double(result1(2)))^2);
error_no(1,ii)=error1;

%% �ع����
[b,bint,r,rint,stats]=regress(lost(a),detect(a,:));

%% �����������
dt=0.01;
t1=x_pos(1)+dt:dt:x_pos(2);
t2=y_pos(1)+dt:dt:y_pos(2);

x_total=int16((x_pos(2)-x_pos(1))/dt)-1;
y_total=int16((y_pos(2)-y_pos(1))/dt)-1;
z=zeros(x_total,y_total);
for i=1:x_total
    for j=1:y_total
        z(i,j)=get_multiFeature([t1(i),t2(j)],5)*b;
    end
end
[ux,uy]=find(z==max(max(z)));
fprintf ('\nthe exact result is (%f,%f)\n ',double(ux*dt+x_pos(1)),double(uy*dt+y_pos(1)));
error2=sqrt((center1(1)-double(ux*dt+x_pos(1)))^2+(center1(2)-double(uy*dt+y_pos(1)))^2);
error_no(2,ii)=error2;

end

t=1:sum_locate;
plot(t,error_no(1,:),'r',t,error_no(2,:),'g');
legend('��ͳ�㷨','ѹ����֪�㷨');
title('�����㷨׼ȷ�ȵıȽ�');
xlabel('����');
ylabel('������/m');

figure;
t=1:sum_locate;
plot(t,error_no(2,:),'g');
title('ѹ����֪�㷨');
xlabel('����');
ylabel('������/m');


%{
mesh(z);

figure;
scatter(detect_pos(a,1),detect_pos(a,2),'r');
hold on;
scatter(max_pos(:,1),max_pos(:,2),'g');
hold on;
scatter(double(ux*dt+x_pos(1)),double(uy*dt+y_pos(1)),'b');
hold on;
scatter(double(result1(1)),double(result1(2)),'k');
%}