function result=sphere(complex_signal,t,x)
%% 功能：生命体征参数匹配值
%% 参数：
%    complex_signal：   %原始的体征信号
%    t：采样点
%    x：输入的未知量矩阵，这里为三维，[a,f,fi]
%    D：输入矩阵的维度
%% 返回值：生命体征参数匹配值
%% 程序主体
Sigma_t=exp(-1j*x(1)*sin(2*pi*x(2)*t+x(3)));
S_sigma=complex_signal.*Sigma_t;
S_spectrum=fft(S_sigma);
result=abs(S_spectrum(1));



