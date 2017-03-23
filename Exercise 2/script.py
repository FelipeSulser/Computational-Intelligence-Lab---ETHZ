import csv
import numpy as np
import re
import matplotlib.pyplot as plt
import gc

from sklearn import cross_validation as cv
import pandas as pd

regex1 = r"\d+"
regex2 = r"c\d+"
X = np.zeros((10000,1000)) #1.000x10.000 matrix

#load data
with open("data_train.csv") as csvfile:
	readCSV = csv.reader(csvfile,delimiter=',')
	next(readCSV) #discard header
	for row in readCSV:
		match = re.search(regex1,row[0])
		match2 = re.search(regex2,row[0])
		row_i = int(match.group(0))-1
		col_i = int(match2.group(0)[1:]) -1
		X[row_i,col_i] = int(row[1])

train_data = X[:10000]
M,N = X.shape
R = np.zeros((10000, 1000))
for i in range(train_data.shape[0]):
    for j in range(train_data.shape[1]):
        R[i,j] = train_data[i,j]


I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

def prediction(U,Z):
    return np.dot(U.T,Z)
def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - prediction(P,Q)))**2)/len(R[R > 0]))

train_errors = []

K = 30
U = np.random.rand(K,M)
Z = np.random.rand(K,N)
print(U.dtype)
n_epochs = 2
lmda = 0.01
gamma = 0.01

#DO SGD
users,movies = R.nonzero()
print(users.shape)
print(movies.shape)


for epoch in range(n_epochs):
    for u, i in zip(users,movies):
       # gamma = 1.0  / (np.sqrt(epoch+1)*10000.0)
        e = R[u, i] - prediction(U[:,u],Z[:,i])
        U[:,u] += gamma * ( +e * Z[:,i] - lmda * U[:,u])
        Z[:,i] += gamma * ( +e * U[:,u] - lmda * Z[:,i])

#now lets predict the data from samplesubmission
fout = open('mysubmission.csv', 'w')
fout.write("Id,Prediction\n")
with open("SampleSubmission.csv") as csvfile:
    readCSV = csv.reader(csvfile,delimiter=',')
    next(readCSV) #discard header
    for row in readCSV:
        match = re.search(regex1,row[0])
        match2 = re.search(regex2,row[0])
        row_i = int(match.group(0))-1
        col_i = int(match2.group(0)[1:]) -1
        calc = prediction(U[:,row_i], Z[:,col_i])
        fout.write(row[0]+","+str(calc)+"\n")
    fout.close()
