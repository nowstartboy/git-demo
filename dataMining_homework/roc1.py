import matplotlib.pyplot as plt

classify=[1,0,1,1,0,1,0,0,0,1] #按顺序记录元组类别，P为1，N为0
n=len(classify)
data=[]
TPR1=[]
FPR1=[]
P=sum(classify)  #正元组的个数
for i in range(0,n):
    TPR=sum(classify[0:i+1])/P #计算第i+1个TPR值e
    FPR=(i+1-sum(classify[0:i+1]))/P #计算第i+1个FPR值
    FPR1.append(FPR)
    data.append([TPR,FPR]) #统一记录TPR,FPR值
print (data)
print (FPR1)
print (TPR1)
#画ROC曲线
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(FPR1,TPR1,'ko-')
ax.set_title('ROC curve')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()



