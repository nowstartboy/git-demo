from math import log

#加载数据，把count放在倒数第二列，类别status放在最后一列
def Dataload():
    dataSet=[['sales', '31-35', '46-50', 30, 'senior'],
             ['sales', '26-30', '26-30', 40, 'junior'],
             ['sales', '31-35', '31-35', 40, 'junior'],
             ['systems', '21-25', '46-50', 20, 'junior'],
             ['systems', '31-35', '66-70', 5, 'senior'],
             ['systems', '26-30', '46-50', 3, 'junior'],
             ['systems', '41-45', '66-70', 3, 'senior'],
             ['marketing', '36-40', '46-50', 10, 'senior'],
             ['marketing', '31-35', '41-45', 4, 'junior'],
             ['secretary', '46-50', '36-40', 4, 'senior'],
             ['secretary', '26-30', '26-30', 6, 'junior']]
    labels=['department','age','salary']
    return dataSet,labels

#求当前分组D的熵
def shannonEnt(dataSet):
    counts=[]
    labelCounts={}
    for featVec in dataSet:
        counts.append(featVec[-2])
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=featVec[-2]    #对每一类别计数(+count)
        else:
            labelCounts[currentLabel]+=featVec[-2]
    numEntries=sum(counts)
    shannonent=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonent-=prob*log(prob,2)
    return shannonent

#选择一属性作为划分点时，从总元组中刨去这一属性对应属性值的列
def splitDataSet(dataSet,axis,value):
#axis 为属性名 value为属性值
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#进行属性选择度量，选择当前最好的分类属性
def chooseBestFeature(dataSet):
    numFeatures=len(dataSet[0])-2
    baseEntropy=shannonEnt(dataSet)
    bestInfoGain=0;
    bestFeatrue=-1;
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueValues=set(featList)
        newEntropy=0
        for value in uniqueValues:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*shannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeatrue=i
    return bestFeatrue

#计算叶结点的多数类
def majorcount(classList):
    classCount={}

    for vote in classList:
        if vote[-1] not in classCount.keys():
            classCount[vote[-1]]=vote[-2]
        else:
            classCount[vote[-1]]=vote[-2]
        sortedClassCount=sorted(classCount.keys(),reverse=True)
    return sortedClassCount[0]

#总函数
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    #分裂结束有两种情况 1.区域内所有元组的类别相同 2.已无分裂属性了
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==2:
        return majorcount(dataSet)
    bestFeat=chooseBestFeature(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree







if __name__=="__main__":
    data,label=Dataload()
#   print (data)
#   print (label)
    '''
    print (shannonEnt(data))
    print (splitDataSet(data,1,'31-35'))
    print (chooseBestFeature(data))

    dataSet=[[10,'yes'],[20,'no'],[11,'yes']];
    print (majorcount(dataSet))
    '''
    data, label = Dataload()
    myTree=createTree(data,label)
    print (myTree)