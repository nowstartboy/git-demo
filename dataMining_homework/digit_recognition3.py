'''
	功能：数字识别
'''
import pandas as pd
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV



def nomalizing(array):
    n=len(array)
    for i in range(n):
        if array[i]!=0:
            array[i]=1
    return array

def loaddata(dataset,dataset_test):
    X_train = dataset.values[0:, 1:]
    Y_train = dataset.values[0:, 0]
    X_test = dataset_test.values[0:, 0:]

    m = len(X_train)
    m_test = len(X_test)
    for i in range(m):
        X_train[i] = nomalizing(X_train[i])

    for j in range(m_test):
        X_test[j] = nomalizing(X_test[j])
    return X_train,Y_train,X_test

def data_pca(X_train,X_test,n=50):
    pca=PCA(n_components=n)
    pca.fit(X_train)
    transform_train=pca.transform(X_train)
    transform_test=pca.transform(X_test)
    return transform_train, transform_test

def Knn_divide(X_train,Y_train,X_test):
    clf=KNeighborsClassifier()
    start = time.clock()
    clf.fit(X_train,Y_train)
    elapsed1 = (time.clock() - start)
    testLabel=clf.predict(X_test)
    elapsed2 = (time.clock() - elapsed1)
    return testLabel,elapsed1,elapsed2

def hand_knn_divide(X_train,Y_train,X_test,k):
    testLabel=[]
    start=time.clock()
    for items in X_test:
        diffMat=np.tile(items,(len(X_train),1))-np.array(X_train)
        sqDiffMat=diffMat**2
        sqDistances=sqDiffMat.sum(axis=1)
        distances=sqDistances**0.5
        sortedDistIndicies=distances.argsort() #返回的数值从小到大的索引
        classCount={}
        for i in range(k):
            voteLabel=Y_train[sortedDistIndicies[i]]
            classCount[voteLabel]=classCount.get(voteLabel,0)+1 #如果classCount中没有voteLabel值，则其为1，否则加一
        sortedClassCount=sorted(classCount.items(),key=lambda d:d[1])
        testLabel.append(sortedClassCount[0][0])
    elapsed1 = (time.clock() - start)
    elapsed2=elapsed1
    return np.array(testLabel),elapsed1,elapsed2




def Network_divide(X_train,Y_train,X_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10), random_state=1)
    start = time.clock()
    clf.fit(X_train, Y_train)
    elapsed1 = (time.clock() - start)
    testLabel = clf.predict(X_test)
    elapsed2 = (time.clock() - elapsed1)
    return testLabel, elapsed1, elapsed2

def Svm_divide(X_train,Y_train,X_test):  #
    svcClf = svm.SVC(C=0.5,kernel='rbf',gamma=0.001)
    start = time.clock()
    svcClf.fit(X_train, Y_train)
    elapsed1 = (time.clock() - start)
    testLabel = svcClf.predict(X_test)
    elapsed2 = (time.clock() - elapsed1)
    return testLabel, elapsed1, elapsed2

def svm_cross_validation(X_train,Y_train):  # SVM调参数
    model=svm.SVC(kernel='rbf')
    param_grid={'C':[1e-1,0.5,1,5,10,20],'gamma':[0.001,0.0001]}
    grid_search=GridSearchCV(model,param_grid,n_jobs=1,verbose=1)
    grid_search.fit(X_train,Y_train)
    best_parameters=grid_search.best_estimator_.get_params()
    for para,val in best_parameters.items():
        print (para,val)

def bag_svm(X_train,Y_train,X_test):
    svcClf = svm.SVC()
    bag_svcClf=BaggingClassifier(svcClf)
    start = time.clock()
    bag_svcClf.fit(X_train, Y_train)
    elapsed1 = (time.clock() - start)
    testLabel = bag_svcClf.predict(X_test)
    elapsed2 = (time.clock() - elapsed1)
    return testLabel, elapsed1, elapsed2

def keras_divide(X_train,Y_train,X_test):
    from keras.models import Sequential
    from keras.layers import  Dense  #BP层
    from keras.layers import Dropout
    model=Sequential()
    model.add(Dense(500,input_dim=784,activitation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10,activation="softmax"))
    model.compile(optimizer='Adagrad',loss='categorical_crossentropy',metrics=['accuracy']) #loss是指目标函数
    model.fit(X_train,Y_train,shuffle=True,nb_epoch=100,batch_size=1000)
    testLabel=model.predict(X_test,batch_size=10000)



def save_results(testLabel):
    result = np.c_[range(1, len(testLabel) + 1), testLabel.astype(int)]
    df_result = pd.DataFrame(result, columns=['ImageId', 'Label'])
    df_result.to_csv('E:\资料文献\数据挖掘课\data_bag_svm.csv', index=False)

def choose_divide(n):
    if n==1:
        testLabel, elapsed1, elapsed2 = Knn_divide(transform_train, Y_train, transform_test) #<1s 77s
    if n==2:
        testLabel, elapsed1, elapsed2 = Network_divide(transform_train, Y_train, transform_test) #9s <1s
    if n==3:
        testLabel, elapsed1, elapsed2 = Svm_divide(transform_train, Y_train, transform_test)  #31s 20s
    if n==4:
        testLabel, elapsed1, elapsed2=hand_knn_divide(transform_train,Y_train,transform_test,5)
    if n==5:
        testLabel, elapsed1, elapsed2 = bag_svm(transform_train, Y_train, transform_test)
    return testLabel, elapsed1, elapsed2

def judge_model(X_train,Y_train,n_train=1000,n_test=1000):
    from  sklearn.metrics import roc_curve,auc
    X_train1=X_train[0:n_train,:]
    Y_train1=Y_train[0:n_train]
    X_train2=X_train[len(X_train)-n_test:,:]
    Y_train2=Y_train[len(X_train)-n_test:]
    Models=[KNeighborsClassifier,MLPClassifier,svm.SVC]
    params=[{},{},{}]
    for Model,param in zip(Models,params):
        start=time.clock()
        total=0
        reg=Model(**param)
        reg.fit(X_train1,Y_train1)
        predictions=reg.predict(X_train2)
        accuracy=(abs(sum(predictions-Y_train2)))/len(predictions)
        print ("accuracy of {0}:{1}".format(Model.__name__,1-accuracy))
        elapsed=(time.clock()-start)
        print ("Cost of time is",elapsed,"s")
    #print ("AUC of {0}:{1}".format(Model._name_,accuracy)) format映射 {0}映射为前一个，{1}映射为后一个






if __name__=="__main__":
    dataset = pd.read_csv("E:\资料文献\数据挖掘课\data2.csv")
    dataset_test = pd.read_csv("E:\资料文献\数据挖掘课\data3.csv")
    X_train, Y_train, X_test=loaddata(dataset,dataset_test)
    transform_train, transform_test=data_pca(X_train,X_test,35)

    #testLabel,elapsed1,elapsed2=Knn_divide(transform_train,Y_train,transform_test)
    testLabel, elapsed1, elapsed2 =choose_divide(5) #1-KNN 2-Network 3-Svm 4-hand-knn 5-bag-svm
    save_results(testLabel)
    print("Training Time used:", int(elapsed1), "s")
    print("Test Time used:", int(elapsed2), "s")
    print (len(X_train[0]))
    print(len(transform_train[0]))
    #print (transform_train[0])

   # judge_model(transform_train,Y_train,30000,10000)
    '''
    #判断PCA 最优降维个数
    for i in [20,30,40,50,60,70,80,90,100]:
        transform_train, transform_test = data_pca(X_train, X_test,i)
        #svcClf = svm.SVC(C=0.5)
        clf = KNeighborsClassifier()
        m = cross_val_score(clf, transform_train, Y_train, cv=10)
        m_avg = sum(m) / len(m)
        print('when n=%d,the score=%f' % (i, m_avg))
    '''
    #svm_cross_validation(transform_train, Y_train) #选择支持向量机最好的参数

