import csv
import numpy
import numpy as np
import pandas as pd
from sklearn import preprocessing
from ml import ml


def load(filename='test.csv'):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        x = []
        y = []

        for line in reader:
            x.append(line[2:-1])
            y.append(line[-1:])

        x = np.array(x[1:])
        y = np.array(y[1:])

        return x, y


def load_transformed(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        x = []
        y = []

        for line in reader:
            x.append(line[0:-1])
            y.append(line[-1:])

        x = np.array(x)
        y = np.array(y)

        return x, y


def load_transformed_1(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        x = []

        for line in reader:
            x.append(line[:-1])

        x = np.array(x)
        return x


def pokpok(array, index, X):
    le = preprocessing.LabelEncoder()
    le.fit(array)
    array = le.transform(array)
    X[:, index] = array


def transform(X):
    jobs = []
    purposes = []
    verifications = []
    status = []
    categories = []
    states = []

    for x in X:
        # month
        x[2] = int(x[2].split(' ')[1])

        # percent
        x[3] = float(x[3][:-1])

        # job
        jobs.append(x[5])

        # years
        x7 = x[6].split(' ')
        if len(x7) == 1:
            x[6] = -1
        elif x7[0] == '<':
            x[6] = 0
        elif "+" in x7[0]:
            x[6] = 11
        else:
            x[6] = x7[0]

        # purpose
        purposes.append(x[7])

        # verification
        verifications.append(x[9])

        # status
        status.append(x[10])

        # categories
        categories.append(x[11])

        # state
        states.append(x[12])

    pokpok(jobs, 5, X)
    pokpok(purposes, 7, X)
    pokpok(verifications, 9, X)
    pokpok(status, 10, X)
    pokpok(categories, 11, X)
    pokpok(states, 12, X)

    return X


def transformY(y):
    for i in range(len(y)):
        if y[i] == 'A':
            y[i] = 1
        elif y[i] == 'B':
            y[i] = 2
        elif y[i] == 'C':
            y[i] = 3
        elif y[i] == 'D':
            y[i] = 4
        elif y[i] == 'E':
            y[i] = 5
        elif y[i] == 'F':
            y[i] = 6
        elif y[i] == 'G':
            y[i] = 7
    return y


def save(X, y):
    output = np.column_stack((X, y))
    output = output.astype(float)
    np.savetxt("test_transformed.csv", output, delimiter=',', fmt="%1.3f")


if __name__ == '__main__':
    x, y = load_transformed("train_transformed.csv")
    X_test = load_transformed_1("test_transformed.csv")
    #x, y = load()
    #x = transform(x)
    #y = transformY(y)
    #save(x, y)
    ml(x, y, X_test)


