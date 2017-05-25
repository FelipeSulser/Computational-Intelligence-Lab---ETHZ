import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.image as mpimg
from tensorflow.python.framework import ops
import scipy.misc
from sklearn import svm
import math
from sklearn import linear_model
import pickle
from sklearn.externals import joblib
import os
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import sparse_encode
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

PATCH_SIZE = 16
CONTEXT_SIZE = 5 # means that for patch i,j we consider the square i-ps*3,j-ps*3 to i+ps*3, j+ps*3
# Create graph
total_window_size = PATCH_SIZE*CONTEXT_SIZE #80x80 in our case

def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            #print(labels[idx][0])
            if labels[idx][0] > 0.5:
                #l = 1
                l = labels[idx][0]
            else:
                #l = 0
                l = labels[idx][0]
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

save_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/sharp_training/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir) 


data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
train_data_filename = data_dir + 'images_shuffled/'
train_labels_filename = data_dir + 'groundtruth_shuffled/' 
pred_filename = (os.path.dirname(os.path.realpath(__file__)))+'/predictions_test/result/'
patch_size = (total_window_size,total_window_size)

num_images = 214


patches = []
for i in range(1, num_images+1):
    imageid = "satImage_%.3d" % i
    image_filename = train_labels_filename + imageid + ".png"
    if os.path.isfile(image_filename):
        print ('Loading ' + image_filename)
        img = mpimg.imread(image_filename)
        img_patches = img_crop(img, total_window_size, total_window_size)
        img_patches = np.asarray(img_patches)
        shaped_data = np.reshape(img_patches,(img_patches.shape[0],-1)) 
        print(shaped_data.shape)
        patches.extend(shaped_data) 
    else:
        print ('File ' + image_filename + ' does not exist')

print("Fetching some atoms")
patches = np.asarray(patches)
n_atoms = int(400/total_window_size)**2
print("Num atoms: "+str(n_atoms))
dico = MiniBatchDictionaryLearning(n_atoms)
patches = np.asarray(patches)
#patches in which format?
V = dico.fit(patches).components_
#need to sparsely encode each patch now with V
num_images = 50
for i in range(1, num_images+1):
    imageid = "prediction_"+str(i)
    image_filename = pred_filename+imageid+".png"
    img = mpimg.imread(image_filename)
    img = rgb2gray(img)
    img_patches = img_crop(img, PATCH_SIZE, PATCH_SIZE)
    img_patches = np.asarray(img_patches)
    new_arr = []
    for x in range(0,img_patches.shape[0]):
        dimx = img_patches[x].shape[0]
        dimy = img_patches[x].shape[1]
        if dimx*dimy == 6400:
            new_arr.append(np.reshape(img_patches[x],(80*80)))
        else:
            datav = np.pad(img_patches[x],((0,80-dimx),(0,80-dimy)),mode='reflect')
            datav = np.reshape(datav,(80*80))
            new_arr.append(datav)

    new_arr = np.asarray(new_arr)
    #shaped_data = np.reshape(img_patches,(1,80*80)) 
    U = sparse_encode(new_arr,V)
    reconstructed = np.dot(U,V)
    img_pred = label_to_img(608, 608, PATCH_SIZE, PATCH_SIZE, reconstructed)
    plt.imshow(img_pred,cmap='gray')
    plt.show()








