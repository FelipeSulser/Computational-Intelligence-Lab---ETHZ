import csv
import numpy as np
import re
import matplotlib.pyplot as plt
import gc
import collections
import pandas as pd
#%matplotlib inline

regex1 = r"\d+"
regex2 = r"c\d+"
X = np.zeros((10000,1000)) #1.000x10.000 matrix


# Load data
with open("../Exercise 1/data_train.csv") as csvfile:
    readCSV = csv.reader(csvfile,delimiter=',')
    next(readCSV) #discard header
    for row in readCSV:
        match = re.search(regex1,row[0])
        match2 = re.search(regex2,row[0])
        row_i = int(match.group(0))-1
        col_i = int(match2.group(0)[1:]) -1
        X[row_i,col_i] = int(row[1])


# returns array of training data and an array of test data
def cv(X, k=5):
    # split indices of nonz-zero values in k sets but randomly!
    nzR, nzC = np.nonzero(X)
    zipped = list(zip(nzR,nzC))
    np.random.shuffle(zipped)
    nzR, nzC = zip(*zipped)
    nzRsplit = np.array_split(nzR, k)
    nzCsplit = np.array_split(nzC, k)
     
    X_train_sets = []
    X_test_sets = []
    for i in range(k):
        X_train = np.copy(X)
        X_train[(nzRsplit[i], nzCsplit[i])] = 0
        #print('X_train has ', np.count_nonzero(X_train), ' non-zeros.')
        X_train_sets.append(X_train)
    
        X_test = np.zeros(X.shape)
        X_test[(nzRsplit[i], nzCsplit[i])] = X[(nzRsplit[i], nzCsplit[i])]
        #print('X_test has ', np.count_nonzero(X_test), ' non-zeros.')
        X_test_sets.append(X_test)
        
    return X_train_sets, X_test_sets

# get RMSE error
def rmse(I,X,Z,U):
    return np.sqrt(np.sum((I * (X - np.dot(U,Z.T)))**2)/len(X[X > 0]))


# Set params and init the algorithm
D,N = X.shape
print('X: ',X.shape)
K = 50
U = np.random.rand(D,K)
Z = np.random.rand(N,K)
n_epochs = 10*(10**6)  #30
lmda = 0.00001       
gamma = 0.0001       # step size

train_errors = collections.defaultdict(list)
test_errors = collections.defaultdict(list)


# SGD with CV
X_train_set, X_test_set = cv(X, k=5)
count = 0
collect_each = 200000
for cv_iter, (X_train, X_test) in enumerate(zip(X_train_set, X_test_set)):
    I = X_train.copy()
    I[I > 0] = 1
    I2 = X_test.copy()
    I2[I2 > 0] = 1
    
    #DO SGD
    ds,ns = X_train.nonzero()
    #print('Nonzeros: ', len(ds))
    #break
    #print 'Users: ',users.shape
    #print 'Movies: ', movies.shape
    print('CV iter = ', cv_iter)
    
    # generate random indeces
    indices = np.random.randint(low=0, high=len(ds), size=n_epochs)
    zipped_ds_ns = list(zip(ds,ns))
    
    for epoch in range(n_epochs):
        if epoch % collect_each == 0:
            print('epoch = ', epoch)
            
        (d, n) = zipped_ds_ns[indices[epoch]]
        #for d, n in zip(ds,ns):
        #gamma = 0.001*(1.0/(np.sqrt(epoch+1.0)))
        gamma = (1e-3)*(1.0/(np.sqrt(epoch/1e6+1.0)))
        #e = R[u, i] - prediction(U[:,u],Z[:,i])
        e = X_train[d, n] - np.dot(U[d,:],Z[n,:].T)
        U[d,:] += gamma * ( e * Z[n,:].T - lmda*U[d,:] )
        Z[n,:] += gamma * ( e * U[d,:] - lmda*Z[n,:] )
        if epoch % collect_each == 0:
            train_rmse = rmse(I,X_train,Z,U) # Calculate root mean squared error from train dataset
            test_rmse = rmse(I2,X_test,Z,U) # Calculate root mean squared error from test dataset
            train_errors[epoch].append(train_rmse)
            test_errors[epoch].append(test_rmse)
            gc.collect()
        
    count += 1


# Collect train and test errors
tr_err = collections.defaultdict(int)
ts_err = collections.defaultdict(int)
for collected_epoch in train_errors.keys():
    tr_err[collected_epoch] = np.mean(train_errors[collected_epoch]) 
    ts_err[collected_epoch] = np.mean(test_errors[collected_epoch]) 


# Viz the error curve
tr_err_lst = sorted(tr_err.items())
ts_err_lst = sorted(ts_err.items())
x, y = zip(*tr_err_lst)
plt.plot(x, y, marker='o', label='Training Data');
x, y = zip(*ts_err_lst)
plt.plot(x, y, marker='v', label='Test Data');
plt.title('SGD Learning Curve')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.savefig('sgd_learning_curve')
plt.show()


# Save the data from samplesubmission
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
        print(calc)
        fout.write(row[0]+","+str(calc)+"\n")
    fout.close()