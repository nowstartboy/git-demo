function result=sphere(complex_signal,t,x)
%% ���ܣ�������������ƥ��ֵ
%% ������
%    complex_signal��   %ԭʼ�������ź�
%    t��������
%    x�������δ֪����������Ϊ��ά��[a,f,fi]
%    D����������ά��
%% ����ֵ��������������ƥ��ֵ
%% ��������
Sigma_t=exp(-1j*x(1)*sin(2*pi*x(2)*t+x(3)));
S_sigma=complex_signal.*Sigma_t;
S_spectrum=fft(S_sigma);
result=abs(S_spectrum(1));



