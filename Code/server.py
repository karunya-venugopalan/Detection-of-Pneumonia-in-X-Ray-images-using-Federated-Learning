import socket 
import threading
import pandas as pd
import numpy as np
import math
import pickle
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

HEADER = 64
PORT = 5050
SERVER = socket.gethostbyname("")
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

clients = set()
clients_lock = threading.Lock()
models = {}

logRegCols = []
for i in range(0,2500):
    s = 'W' + str(i)
    logRegCols.append(s)
logRegDF = pd.DataFrame(columns=logRegCols)

knnCols = []
for i in range(0,50):
    s = 'PCA' + str(i)
    knnCols.append(s)
knnCols.append('Y')
knnDF = pd.DataFrame(columns = knnCols)

svmCols = []
for i in range(0,2500):
    s = 'param' + str(i)
    svmCols.append(s)
svmDF = pd.DataFrame(columns = svmCols)

def logisticRegression():
    wAvg = [mean(logRegDF[i]) for i in list(logRegDF.columns)]
    return wAvg


def findBestK():
    x = knnDF.iloc[:,0:50]
    y = knnDF['Y']
    y=y.astype('int')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    k = [1,3,5]
    kscore = {}
    for i in k:
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(x_train,y_train)
        score = knn.score(x_test, y_test)
        kscore[i] = score
    bestk = max(kscore, key=kscore.get)
    return bestk

def SVM():
    svmAverage = [mean(svmDF[i]) for i in list(svmDF.columns)]
    return svmAverage

def handle_client(conn, addr):
    print("NEW CONNECTION ", addr," connected.")
    connected = True
    with clients_lock:
        clients.add(conn)
    while connected:
        msg = conn.recv(HEADER).decode(FORMAT)
        print("Message from client:", msg)
        if msg == 'Logistic Regression':
            logReg = conn.recv(1048576)
            logRegData = pickle.loads(logReg)
            logRegDF.loc[len(logRegDF)] = logRegData
            if (len(logRegDF) == 3):
                wAvg = logisticRegression()
                send_data = pickle.dumps(wAvg)
                with clients_lock:
                    for c in clients:
                        c.sendall(send_data)

        if msg == 'KNN':
            knn = conn.recv(1048576)
            knnPCA = pickle.loads(knn)  
            global knnDF 
            knnDF = knnDF.append(knnPCA)   
            if(len(knnDF) == 4949):
                bestK = findBestK()
                send_data = pickle.dumps(bestK)
                with clients_lock:
                    for c in clients:
                        c.sendall(send_data)
        
        if msg == 'SVM':
            svm = conn.recv(1048576)
            svmParams = pickle.loads(svm)
            svmDF.loc[len(svmDF)] = svmParams
            if (len(svmDF) == 3):
                svmAvg = SVM()
                send_data = pickle.dumps(svmAvg)
                with clients_lock:
                     for c in clients:
                         c.sendall(send_data)

        if 'Best Model: ' in msg:
            global models
            model = msg.split(': ')
            model = model[1]
            if model in models.keys():
                models[model] += 1
            else:
                models[model] = 1
            vals = sum(list(models.values()))
            if vals == 3:
                print("Best Model: ", max(models, key=models.get))

        if msg == DISCONNECT_MESSAGE:
            connected = False
        if connected == False:
            break
               
    conn.close()
        

def start():
    server.listen(5)
    print("[LISTENING] Server is listening on ",server)
    print("")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()


print("[STARTING] server is starting...")
start()