# coding=utf-8
import numpy as np
import time,types,os
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs

# 导入分类器
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# 分类器配置
classifiers = {
    'KN': KNeighborsClassifier(3),
    'SVM':svm.LinearSVC(C=1),
    'DT': DecisionTreeClassifier(max_depth=5),
    'RF': RandomForestClassifier(n_estimators=10, max_depth=5, max_features=1),  # clf.feature_importances_
    'ET': ExtraTreesClassifier(n_estimators=10, max_depth=None),  # clf.feature_importances_
    'GB': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), # clf.feature_importances_
    'NB': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'QD': QuadraticDiscriminantAnalysis()}

# 创建标签文件，共9类，每类1990个文件 ，分别用数字0~8表示   
label=np.zeros(17910)
base0=np.zeros(1990)
base1=np.ones(1990)
label[:1990]=base0
label[1990:3980]=base1
label[3980:5970]=2*base1
label[5970:7960]=3*base1
label[7960:9950]=4*base1
label[9950:11940]=5*base1
label[11940:13930]=6*base1
label[13930:15920]=7*base1
label[15920:]=8*base1   

# 加载生成的文本向量后进行分类
output_file=open('tf-idf_LSI_classification_record.txt','a') #记录分类结果
for dimensions in [50,100,200,500]:
    data_name=('train_LSI_vector_'+str(dimensions)+'.npy')
    datasets=np.load(data_name) #加载不同维度的文本向量
    for name,clf in classifiers.items():
        begin=time.time() #记录分类所用的时间
        scores=cross_val_score(clf,datasets,label) #交叉验证分类准确率
        # 记录分类结果和分类所需的时间
        output_file.write('vector dimensions=%s '%str(dimensions))
        print 'LSI vector dimensions=%s' %str(dimensions)
        output_file.write('with %s classifier '%name)
        print 'with %s classifier' %name
        output_file.write('Precision=%s'%scores.mean())
        print 'Precision=%s' %scores.mean()
        output_file.write('need %s seconds\n'%str(time.time()-begin))
        print 'take %s seconds\n' %str(time.time()-begin)
