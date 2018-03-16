clear;  
clc;  
mu1 = [1 2 ];  
Sigma1 = [1 0 ;0 1 ];  
mu2 = [-4 2 ];  
Sigma2 = [1 0;0 1 ];  
rng(1); % For reproducibility  
X = [mvnrnd(mu1,Sigma1,1000);mvnrnd(mu2,Sigma2,1000)]; % 2000 x 2  
GMModel = fitgmdist(X,2); %fit GMM distribution  
  
figure;  
y = [zeros(1000,1);ones(1000,1)];  
hold on;  
ezsurf (@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}));  
title('{\bf 3-D Fitted Gaussian Mixture}');  
hold off;  
  
figure;  
y = [zeros(1000,1);ones(1000,1)];  
h = gscatter(X(:,1),X(:,2),y);  
hold on;  
ezcontour (@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}));  
title('{\bf Fitted Gaussian Mixture Contours}');  
legend(h,'Model 0','Model1')  
hold off;  