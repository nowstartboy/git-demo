import numpy as np
age=[13,15,16,16,19,20,20,21,22,22,25,25,25,25,30,33,33,35,35,35,35,36,40,45,46,52,70]
age_min=min(age)
age_max=max(age)
new_min=0
new_max=1
num=35
v1=(num-age_min)*(new_max-new_min)/(age_max-age_min)+new_min
age_mean=np.mean(age)
v2=(num-age_mean)/12.94
v3=num/100
print('最小最大规范：',v1)
print('z分数规范：', v2)
print('小数定标：',v3)






