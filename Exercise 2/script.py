import csv
import numpy as np
import re
import matplotlib.pyplot as plt

def prediction(U,Z):
    return np.dot(U.T,Z)

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

print('Done parsing\n')

K = 100
M,N = X.shape
U = np.random.rand(K,M)
Z = np.random.rand(K,N)
n_epochs = 20
lmda = 0.01 #regularizer
gamma = 0.01 #learning rate

#DO SGD
users,movies = X.nonzero()
print(users.shape)
print(movies.shape)

for epoch in range(n_epochs):
    for u, i in zip(users,movies):
        e = X[u, i] - prediction(U[:,u],Z[:,i])
        U[:,u] += gamma * ( e * Z[:,i] - lmda * U[:,u])
        Z[:,i] += gamma * ( e * U[:,u] - lmda * Z[:,i])  

print('SGD finished')
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