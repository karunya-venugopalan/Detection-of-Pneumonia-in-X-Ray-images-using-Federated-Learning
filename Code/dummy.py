import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
# Loading the data
train = pd.read_csv('C://Users//Karunya V//Documents//sem5//ML lab//Package//chest_xray//data//client1_train.csv')
test = pd.read_csv('C://Users//Karunya V//Documents//sem5//ML lab//Package//chest_xray//data//client1_test.csv')
x = train.iloc[:,3:2502]
y = train['Diagnosis']
xtest = test.iloc[:,3:2502]
ytest = test['Diagnosis']
def pca():
    Cols = []
    for i in range(0,50):
        s = 'PCA' + str(i)
        Cols.append(s)
    #knnCols.append('Y')
    pca = PCA(n_components=50)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns=Cols)
    principalDf['Y'] = y
    return principalDf

pca_svm = pca()
svmx = pca_svm.iloc[:,0:50]
svmy = pca_svm['Y']
print(type(svmy)," ",svmy.shape)
print(type(svmx)," ",svmx.shape)
clf = SVC(kernel='linear')
print("HIII")
clf.fit(svmx, svmy)
print("HI")

print(len(clf.coef_[0]))
print(clf.intercept_)