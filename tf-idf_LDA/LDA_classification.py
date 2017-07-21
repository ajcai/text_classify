# coding=utf-8
import numpy as np
import time,types,os
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs

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


classifiers = {
    'KN': KNeighborsClassifier(3),
    'SVM':svm.LinearSVC(C=1),
    'RF': RandomForestClassifier(n_estimators=10, max_depth=5, max_features=1),  # clf.feature_importances_
    'GB': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), # clf.feature_importances_
    'NB': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis()}

# load data    
label=np.zeros(17910) #creat label
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

record_file=open('tf-idf_LDA_test_record.txt','w')
for dimensions in [50,100,200,500,600]:
    data_name=('train_LDA_vector_'+str(dimensions)+'.npy')
    datasets=np.load(data_name)
    print 'test features with %s dimensions'%dimensions
    
    for name, clf in classifiers.items():
        begin=time.time()
        scores = cross_val_score(clf, datasets,label)
        print name,'\t--> ',scores.mean()
        #print scores,scores.mean()
        record_file.write('vector dimensions=%s'%str(dimensions))
        record_file.write(' with %s  classifier'%name)


        record_file.write(' precision=%s'%str(scores.mean()))
        print time.time() - begin,'seconds used for classification'
        record_file.write(' need %s seconds\n'%str(time.time()-begin))
        print '--------------------------------------------------------------------------'
    print 'test result has saved to tf-idf_LDA_test_record.txt'
