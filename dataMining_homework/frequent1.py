'''
时间：2016.11.4
功能：Apriori算法生成频繁项集
'''
#加载数据集
def loadDataSet():
    return [['M','O','N','K','E','Y'],['D','O','N','K','E','Y'],['M','A','K','E'],['M','U','C','K','Y'],['C','O','O','K','I','E']]

#利用原始数据集生成C1
def createC1(dataSet):
    C1=[]
    for TID in dataSet:
        for item in TID:
            if item not in C1:
                C1.append(item)
    C1.sort()
    return set(C1)

#根据Ck生成Lk
def create_Lk(Data,Ck,minSupport):
    ssCnt={}
    ssCnt_cut={}        #储存Lk
    pass_ssCnt=[]       #储存Lk形成时不满足支持度的枝节，以用于生成Ck+1时剪枝的操作
    for TID in Data:
        for no in Ck:
            if set(no).issubset(TID):
                if no not in ssCnt:
                    ssCnt[no]=1
                else:
                    ssCnt[no]+=1
    TID_num=len(Data)
    for key in ssCnt:
        support=ssCnt[key]/TID_num
        if support<minSupport:
            pass_ssCnt.append(set(key))
        else:
            ssCnt_cut[key]=ssCnt[key]
    return ssCnt_cut,pass_ssCnt

#根据Lk 生成Ck+1
def aprioriGen(Lk,k,pass_ssCnt):
    Ckplus=[]
    Lk_key1=list(Lk.keys())
    Lk_key=[sorted(key) for key in Lk_key1]
    for i in range(len(Lk)):
        for j in range(i+1,len(Lk)):
            L1=Lk_key[i][:k-2]
            L2=Lk_key[j][:k-2]
            if L1==L2:
                judge=1         #判定是否需要被剪掉
                c=set(Lk_key[i]+Lk_key[j])
                for freq in pass_ssCnt:   #剪枝
                    if freq.issubset(c):
                        judge=0
                if judge==1:
                    Ckplus.append(tuple(c))
    return Ckplus

#总函数，生成频繁项集
def apriori(dataSet,minSupport=0.5):
    L=[]
    C1=createC1(dataSet)
    L1,pass_ssCnt=create_Lk(dataSet,C1,minSupport)
    L.append(L1)
    k=2
    while(len(L[k-2])>0):
        Ck=aprioriGen(L[k-2],k,pass_ssCnt)
        Lk,pass_ssCnt=create_Lk(dataSet,Ck,minSupport)
        L.append(Lk)
        k+=1
    return L

'''
时间：2016.11.6
功能：FP-Growth
'''
#FP树的结点类
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name=nameValue   #结点名称
        self.count=numOccur   #结点的支持度计数
        self.nodeLink=None   #链接名称相同的结点项
        self.parent=parentNode  #父结点
        self.children={}        #子结点
    def inc(self,numOccur):    #计数加1
        self.count+=numOccur
    def disp(self,ind=1):      #打印树
        print (' '*ind,self.name,' ',self.count)
        for child in self.children.values():
            child.disp(ind+1)

#建立总的FP树
def createTree(dataSet,minSup=1):
    headerTable={}    #原始数据集信息
    headerTable_cut={}  #筛选后的数据集信息
    headerTable1={}     #筛选后的数据集的计数信息+链接信息
    for trans in dataSet:
        for item in trans:
            headerTable[item]=headerTable.get(item,0)+dataSet[trans]
    for k in headerTable:
        if headerTable[k]<minSup:
            pass
        else:
            headerTable_cut[k]=headerTable[k]
            headerTable1[k]=[headerTable[k],None]
    itemSet=set(headerTable_cut.keys())
    if len(itemSet)==0:
        return None,None
    retTree=treeNode('Null Set',1,None)
    for tranSet,count in dataSet.items():
        localD={}
        for item in tranSet:
            if item in itemSet:
                localD[item]=headerTable_cut[item]
        if len(localD)>0:
            orderedItems=[v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
            updateTree(orderedItems,retTree,headerTable1,count)
    return retTree,headerTable1

#往树中添加某条事务中符合支持度的数据
def updateTree(items,inTree,headerTable,count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]]=treeNode(items[0],count,inTree)
    if headerTable[items[0]][1]==None:
        headerTable[items[0]][1] = inTree.children[items[0]]
    elif headerTable[items[0]][1]==inTree.children[items[0]]:
        pass
    else:
        updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
    if len(items)>1:
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)

#更新链接信息
def updateHeader(nodeToTest,targetNode):
    while(nodeToTest.nodeLink!=None):
        nodeToTest=nodeToTest.nodeLink
    nodeToTest.nodeLink=targetNode

#得到树中某结点回溯到头结点的路径
def ascendTree(leafNode,prefixPath):
    if leafNode.parent!=None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

#得到某元素的条件FP树
def findPrefixPath(basePat,treeNode):  #取出basePat的FP树
    condPats={}
    while treeNode!=None:
        prefixPath=[]
        ascendTree(treeNode,prefixPath)
        if len(prefixPath)>1:
            condPats[frozenset(prefixPath[1:])]=treeNode.count
        treeNode=treeNode.nodeLink
    return condPats

#根据FP树挖掘频繁项集的总函数
def mineTree(inTree,headerTable,minSup,preFix,itemList):
    headerTable2={}
    for k in headerTable:
        headerTable2[k]=headerTable[k][0]
    bigL=[v[0] for v in sorted(headerTable2.items(),key=lambda p:p[1])]
    for basePat in bigL:
        newFreqSet=preFix.copy()
        newFreqSet.add(basePat)
        itemList.append(newFreqSet)
        print (itemList)
        condPattBases=findPrefixPath(basePat,headerTable[basePat][1])
        myCondTree,myHead=createTree(condPattBases,minSup)
        if myHead!=None:
            mineTree(myCondTree,myHead,minSup,newFreqSet,itemList)

'''
时间：2016.11.10
功能：从频繁项集中挖掘出规则
'''
#生成规则的总函数
def generateRules(L,minConf=0.7):
    bigRuleList=[]
    supportData={}
    for k in range(0,len(L)):
        for u in L[k]:
            u1=frozenset(u)
            supportData[u1]=L[k][u]

    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1=[item for item in freqSet]
            if (i>1):
                calcConf(set(freqSet), H1, supportData, bigRuleList, minConf)
                rulesFromConseq(set(freqSet),H1,supportData,bigRuleList,minConf)
            else:
                calcConf(set(freqSet),H1,supportData,bigRuleList,minConf)
    return bigRuleList

#判定当前规则项是否符合置信度
def calcConf(freqSet,H,supportData,br1,minConf=0.7):
    prunedH=[]       #输出符合置信度的规则
    pass_cut=[]      #记录不符合置信度的规则后件，以用于剪枝
    for conseq in H:
        conseq=set(conseq)
        conf=supportData[frozenset(freqSet)]/supportData[frozenset(freqSet-conseq)]
        if conf>=minConf:
            print (freqSet-conseq,'-->',conseq,'  conf:',conf,'  sup:',supportData[frozenset(freqSet)]/5)
            br1.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
        else:
            pass_cut.append(conseq)
    return prunedH,pass_cut

#产生新的规则项，以供calcConf函数进行判断
def rulesFromConseq(freqSet,H,supportData,br1,minConf=0.7,pass_cut=[]):
    m=len(H[0])
    H1={}
    if (len(freqSet)>(m+1)):
        for h_value in H:
            H1[tuple(h_value)]=1
        Hmp1=aprioriGen(H1,m+1,pass_cut)
        Hmp1,pass_cut=calcConf(freqSet,Hmp1,supportData,br1,minConf)
        if (len(Hmp1)>1):
            rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf,pass_cut)






if __name__=="__main__":
    '''
    #测试Aporior频繁项集算法
    print ('start')
    dataSet=loadDataSet()
    #print (dataSet)
    C1=createC1(dataSet)
    print (C1)
    L1,pass_ssCnt=create_Lk(dataSet,C1,0.6)
    print(L1)
    print(pass_ssCnt)
    C2=aprioriGen(L1,2,pass_ssCnt)
    print ('sffds')
    print (C2)
    L2,pass_ssCnt=create_Lk(dataSet,C2,0.6)
    print(L2)
    '''
    dataSet=loadDataSet()
    L=apriori(dataSet,0.6)
    print (L)

    #生成规则
    bigRuleList=generateRules(L,0.8) #L为Apriori算法所产生的频繁项集




    '''
    #测试FP树挖掘频繁项
    dataSet=loadDataSet()
    dataSet1={}
    for k in dataSet:
        dataSet1[frozenset(k)]=1
    #print (dataSet1)
    retTree, headerTable1=createTree(dataSet1)
    #print ('en')
    #print (headerTable1)
    #condPats=findPrefixPath('M',headerTable1['M'][1])
    #print (condPats)
    #print ('haoba')
    #print (condPats)
    #print (headerTable1.items())

    freq=[]
    itemList=mineTree(retTree,headerTable1,3,set([]),freq)
    print (freq)
    '''











