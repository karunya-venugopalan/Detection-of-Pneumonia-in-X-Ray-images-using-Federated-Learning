import socket
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import SGDClassifier



HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
SERVER = "192.168.1.112"     
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)


# Loading the data
train = pd.read_csv('C://Users//Karunya V//Documents//sem5//ML lab//Package//chest_xray//data//client2_train.csv')
test = pd.read_csv('C://Users//Karunya V//Documents//sem5//ML lab//Package//chest_xray//data//client2_test.csv')
x = train.iloc[:,3:2502]
y = train['Diagnosis']
xtest = test.iloc[:,3:2502]
ytest = test['Diagnosis']
scores = {}


def sigmoid(z):
    return (1/(1+np.exp(-z)))


def sendString(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(message)



def sendLogisticRegression(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(message)
    sgdc = SGDClassifier(max_iter=1000, tol=0.01, loss='log')
    sgdc.fit(x, y)
    params_list =  list(sgdc.coef_[0])
    params_list.append(sgdc.intercept_[0])
    send_data = pickle.dumps(params_list)
    client.send(send_data)
    print("Logistic regression SENT")


def ModelLogReg():
    d = client.recv(1048576)
    wFinal = pickle.loads(d)
    print("Logistic regression parameters recieved client2 ")
    ypred = []
    w = wFinal[0:-1]
    b = wFinal[-1]
    for i in xtest.values:
        ypred.append(int(sigmoid(np.dot(w,i) + b) >= 0.5))
    lr_f1 = metrics.f1_score(ytest, ypred,average="macro")
    scores['Logistic Regression'] = lr_f1
    print("Logistic Regression Accuracy:",metrics.accuracy_score(ytest, ypred))
    print("Logistic Regression Precision:",metrics.precision_score(ytest, ypred,average="macro",zero_division=0))
    print("Logistic Regression Recall:",metrics.recall_score(ytest, ypred,average="macro"))
    print("Logistic Regression F1-Score:",metrics.f1_score(ytest, ypred,average="macro"))
    print("\n")
    

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

def sendKNN(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(message)
    pca_val = pca()
    send_data = pickle.dumps(pca_val)
    client.send(send_data)
    print("knn SENT")

def ModelKNN():
    d = client.recv(1048576)
    kFinal = pickle.loads(d)
    print("Best K recieved at client2: ",kFinal)
    knn = KNeighborsClassifier(n_neighbors = kFinal)
    knn.fit(x,y)
    ypred = knn.predict(xtest)
    knn_f1 = metrics.f1_score(ytest, ypred,average="macro")
    scores['KNN'] = knn_f1
    print("Knn Accuracy:",metrics.accuracy_score(ytest, ypred))
    print("Knn Precision:",metrics.precision_score(ytest, ypred,average="macro",zero_division=0))
    print("Knn Recall:",metrics.recall_score(ytest, ypred,average="macro"))
    print("Knn F1-Score:",metrics.f1_score(ytest, ypred,average="macro"))
    print("\n")

def sendSVM(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(message)
    clf = SVC(kernel='linear')
    clf.fit(x, y)
    params_list = list(clf.coef_[0])
    params_list.append(clf.intercept_[0])
    send_data = pickle.dumps(params_list)
    client.send(send_data)
    print("SVM SENT")

def ModelSVM():
    d = client.recv(1048576)
    wFinal = pickle.loads(d)
    print("SVM parameters recieved at client2")
    w = wFinal[0:-1]
    b = wFinal[-1]
    ypred = []
    for i in xtest.values:
        if int(np.dot(w,i)+b) >= 1:
            ypred.append(1)
        else:
            ypred.append(0)
    svm_f1 = metrics.f1_score(ytest, ypred,average="macro")
    scores['SVM'] = svm_f1
    print("SVM Accuracy:",metrics.accuracy_score(ytest, ypred))
    print("SVM Precision:",metrics.precision_score(ytest, ypred,average="macro",zero_division=0))
    print("SVM Recall:",metrics.recall_score(ytest, ypred,average="macro"))
    print("SVM F1-Score:",metrics.f1_score(ytest, ypred,average="macro"))
    print("\n")

sendString("Hello World! from Client 2")
sendLogisticRegression("Logistic Regression")
ModelLogReg()
sendKNN("KNN")
ModelKNN()
sendSVM("SVM")
ModelSVM()
bestModel = max(scores, key=scores.get)
ans = 'Best Model: ' + bestModel
sendString(ans)
sendString(DISCONNECT_MESSAGE)