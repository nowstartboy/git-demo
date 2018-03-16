function [yn,W,en]=LMS(xn,dn,M,mu,itr)
% LMS(Least Mean Squre)�㷨
% �������:
%     xn   ������ź�����      (������)
%     dn   ����������Ӧ����    (������)
%     M    �˲����Ľ���        (����)
%     mu   ��������(����)      (����)     Ҫ�����0,С��xn����ؾ����������ֵ�ĵ���    
%     itr  ��������            (����)     Ĭ��Ϊxn�ĳ���,M<itr<length(xn)
% �������:
%     W    �˲�����Ȩֵ����     (����)
%          ��СΪM x itr,
%     en   �������(itr x 1)    (������)  
%     yn   ʵ���������             (������)

% ������������Ϊ4����5��
if nargin == 4                 % 4��ʱ�ݹ�����Ĵ���Ϊxn�ĳ��� 
    itr = length(xn);
elseif nargin == 5             % 5��ʱ����M<itr<length(xn)
    if itr>length(xn) || itr<M
        error('��������������С!');
    end
else
    error('������������ĸ���!');
end


% ��ʼ������
en = zeros(itr,1);             % �������,en(k)��ʾ��k�ε���ʱԤ�������ʵ����������
W  = zeros(M,1);             % ÿһ�д���һ����Ȩ����,ÿһ�д���-�ε���,��ʼΪ0
yn=zeros(size(xn));
yn(1:M)=xn(1:M);

% ��������
for k = M:itr                  % ��k�ε���
    x = xn(k:-1:k-M+1);        % �˲���M����ͷ������
    yn(k) = W.' * x;        % �˲��������
    en(k) = dn(k) - yn(k) ;        % ��k�ε��������
    
    % �˲���Ȩֵ����ĵ���ʽ
    W = W + 2*mu*en(k)*x;
end

% ������ʱ�˲������������
% yn = inf * ones(size(xn));
% yn(1:M)=xn(1:M);
% for k = M:length(xn)
%     x = xn(k:-1:k-M+1);
%     yn(k) = W.'* x;
% end