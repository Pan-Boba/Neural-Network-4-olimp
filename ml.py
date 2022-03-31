from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import csv


def softmax(x):
    x = x/np.max(x)
    exp_sum = np.sum(np.exp(x))
    for i in range(len(x)):
        x[i] = float(np.exp(x[i])/exp_sum)
    return x


def delete_ass(x):
    x = np.delete(x, 5, 1)
    x = np.delete(x, 27, 1)
    return x


def _normalize(x):
    x = x.astype(float)
    x = x.T
    for i in range(len(x)):
        x[i] = softmax(x[i])
    x = x.T
    return x


def scale(x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x


def ml(x, y, X_test):
    x = delete_ass(x)
    X_test = delete_ass(X_test)
    x = _normalize(x)
    X_test = _normalize(X_test)
    x = scale(x)
    X_test = scale(X_test)
    x = pca(x)
    X_test = pca(X_test)


    model = svm.SVC(C=100, gamma=0.1)
    model.fit(x, y)
    y_pred = model.predict(X_test)
    print(y_pred)

    #y_pred = y_pred.astype(float)
    for i in range(len(y_pred)):
         if float(y_pred[i]) == 1:
             y_pred[i] = 'A'
         elif float(y_pred[i]) == 2:
             y_pred[i] = 'B'
         elif float(y_pred[i]) == 3:
             y_pred[i] = 'C'
         elif float(y_pred[i]) == 4:
             y_pred[i] = 'D'
         elif float(y_pred[i]) == 5:
             y_pred[i] = 'E'
         elif float(y_pred[i]) == 6:
             y_pred[i] = 'F'
         elif float(y_pred[i]) == 7:
             y_pred[i] = 'G'
    with open('final.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(y_pred)


def pca(x):
    pca = PCA(n_components=15)
    proj = pca.fit_transform(x)
    return proj
