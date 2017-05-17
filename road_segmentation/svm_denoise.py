# Nonlinear SVM Example
#----------------------------------
#
# This function wll illustrate how to
# implement the gaussian kernel on
# the iris dataset.
#
# Gaussian Kernel:
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.image as mpimg
from tensorflow.python.framework import ops
import scipy.misc
from sklearn import svm
from sklearn import linear_model
import pickle
from sklearn.externals import joblib
import os

#ops.reset_default_graph()

TRAIN = True
PATCH_SIZE = 16
CONTEXT_SIZE = 3 # means that for patch i,j we consider the square i-ps*3,j-ps*3 to i+ps*3, j+ps*3
# Create graph
total_pixel_length = PATCH_SIZE+2*CONTEXT_SIZE*PATCH_SIZE
#sess = tf.Session()


if TRAIN:
    real_y = []
    train_x = []
    max_val= 0
    train_img_path = (os.path.dirname(os.path.realpath(__file__)))+"/training/groundtruth/"
    label_img_path = (os.path.dirname(os.path.realpath(__file__)))+"/training/groundtruth/"
    for i in range(1, 101):
        imageid = "satImage_%.3d" % i
        #imageid = "prediction_"+str(i)
        image_filename = train_img_path +imageid+ ".png"
        img = mpimg.imread(image_filename)

        #2D matrix between 0,1
        num_patches = int(img.shape[0]/PATCH_SIZE)
        for i in range(CONTEXT_SIZE,num_patches-CONTEXT_SIZE):
            for j in range(CONTEXT_SIZE, num_patches-CONTEXT_SIZE):
                curr_x = img[(i*PATCH_SIZE - CONTEXT_SIZE*PATCH_SIZE):(i*PATCH_SIZE+CONTEXT_SIZE*PATCH_SIZE),(j*PATCH_SIZE-CONTEXT_SIZE*PATCH_SIZE):(j*PATCH_SIZE+CONTEXT_SIZE*PATCH_SIZE)]
                max_val = max(max_val,(i*PATCH_SIZE+CONTEXT_SIZE*PATCH_SIZE)-(i*PATCH_SIZE - CONTEXT_SIZE*PATCH_SIZE))
                flattened = curr_x.flatten()
                flattened = flattened/(np.linalg.norm(flattened)+1)
                train_x.append(flattened)
                meanval = np.mean(img[i*PATCH_SIZE:i*PATCH_SIZE+PATCH_SIZE,j*PATCH_SIZE:j*PATCH_SIZE+PATCH_SIZE])
                if meanval > 0.5:
                    real_y.append(1)
                else:
                    real_y.append(0)



    print("Finished loading images")
    print("Computing SVM with RBF kernel")

    clf = svm.SVC(C=1.0,kernel='rbf')
    clf.fit(train_x,real_y)
    joblib.dump(clf, 'svcmodel.pkl') 
else:
    clf = svm.SVC(C=1.0,kernel='rbf')
    clf = joblib.load('svcmodel.pkl')
    #now predict
    predict_img_path = "/Users/felipesulser/Dropbox/ETH/CIL/TensorFlow/predictions_test/result/"
    output_img_path = "/Users/felipesulser/Dropbox/ETH/CIL/TensorFlow/predictions_test/result_denoised/"
    print("Predicting patches")
    for i in range(1,51):
        imageid = "prediction_"+str(i)
        image_filename = predict_img_path+imageid+".png"
        img = mpimg.imread(image_filename)

        num_patches = int(img.shape[0]/PATCH_SIZE)
        for i in range(CONTEXT_SIZE,num_patches-CONTEXT_SIZE):
            for j in range(CONTEXT_SIZE, num_patches-CONTEXT_SIZE):
                curr_x = img[(i*PATCH_SIZE - CONTEXT_SIZE*PATCH_SIZE):(i*PATCH_SIZE+CONTEXT_SIZE*PATCH_SIZE),(j*PATCH_SIZE-CONTEXT_SIZE*PATCH_SIZE):(j*PATCH_SIZE+CONTEXT_SIZE*PATCH_SIZE)]
                flattened = curr_x.flatten()
                flattened = flattened/(np.linalg.norm(flattened)+1)
                flattened = flattened.reshape(1,-1)
                meanval = np.mean(img[i*PATCH_SIZE:i*PATCH_SIZE+PATCH_SIZE,j*PATCH_SIZE:j*PATCH_SIZE+PATCH_SIZE])
                #print("---------")
                #print("WE HAVE : "+str(meanval))
                predic = clf.predict(flattened)
                img[i*PATCH_SIZE:i*PATCH_SIZE+PATCH_SIZE,j*PATCH_SIZE:j*PATCH_SIZE+PATCH_SIZE] = (meanval+predic)/2
                #print("EXPECTED: "+str(predic))
        save_str = output_img_path+imageid+".png"
        scipy.misc.imsave(save_str,img)


#get output data



