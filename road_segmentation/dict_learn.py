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
CTX_SIZE = 2 # means xx i xx so 5

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

def img_crop_context(im, w, h,context_factor, sub_mean=False):

    padding_type = 'reflect'
    cf = context_factor
    is_2d = len(im.shape) < 3
    if is_2d:
        padded_img = np.pad(im, cf, padding_type)
    else:
        padded_img = np.pad(im, ((cf,cf),(cf,cf),(0,0)), padding_type)

    list_patches = []
    imgheight = padded_img.shape[0]
    imgwidth = padded_img.shape[1]
 
    for i in range(cf,imgheight-cf,h):
        for j in range(cf,imgwidth-cf,w):
            im_patch = np.zeros(1)
            
            if is_2d:
                im_patch = padded_img[i-cf:i+h+cf, j-cf:j+w+cf]
                if im_patch.shape[0] < 2*cf+h and im_patch.shape[1] == 2*cf+w:
                    pad_size = 2*cf+h - im_patch.shape[0]
                    im_patch = np.pad(im_patch, ((0,pad_size),(0,0) ), padding_type)
                    
                elif im_patch.shape[1] < 2*cf+w and im_patch.shape[0] == 2*cf+h:
                    pad_size = 2*cf+w - im_patch.shape[1]
                    im_patch = np.pad(im_patch, ((0,0),(0,pad_size)), padding_type)
                    
                elif im_patch.shape[1] < 2*cf+w and im_patch.shape[0] < 2*cf+h:
                    pad_size0 = 2*cf+h - im_patch.shape[0]
                    pad_size1 = 2*cf+w - im_patch.shape[1]
                    im_patch = np.pad(im_patch, (( 0,pad_size0),(0,pad_size1)), padding_type)
                    
            else:
                im_patch = padded_img[i-cf:i+h+cf, j-cf:j+w+cf, :]
                if im_patch.shape[0] < 2*cf+h and im_patch.shape[1] == 2*cf+w:
                    pad_size = 2*cf+h - im_patch.shape[0]
                    im_patch = np.pad(im_patch, ((0,pad_size),(0,0) ,(0,0)), padding_type)
                    
                elif im_patch.shape[1] < 2*cf+w and im_patch.shape[0] == 2*cf+h:
                    pad_size = 2*cf+w - im_patch.shape[1]
                    im_patch = np.pad(im_patch, ((0,0),(0,pad_size), (0,0)), padding_type)
                    
                elif im_patch.shape[1] < 2*cf+w and im_patch.shape[0] < 2*cf+h:
                    pad_size0 = 2*cf+h - im_patch.shape[0]
                    pad_size1 = 2*cf+w - im_patch.shape[1]
                    im_patch = np.pad(im_patch, (( 0,pad_size0),(0,pad_size1),(0,0)), padding_type)
                         
            list_patches.append(im_patch)

    return list_patches
def binarize(img,block_size,threshold):
    
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    
    numblockwidth = int(imgwidth/block_size)

    numblockheight = int(imgheight/block_size)

    for i in range(0,numblockwidth):
        for j in range(0, numblockheight):
            pixel_i = i*block_size
            pixel_j = j*block_size
            avg = np.mean(img[pixel_i:pixel_i+block_size,pixel_j:pixel_j+block_size])
            if(avg > threshold):
                img[pixel_i:pixel_i+block_size,pixel_j:pixel_j+block_size] = 1
            else:
                img[pixel_i:pixel_i+block_size,pixel_j:pixel_j+block_size] = 0

    return img

def mean_img_per_patch(img,block_size):
    
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    
    numblockwidth = int(imgwidth/block_size)

    numblockheight = int(imgheight/block_size)
    newimg = np.zeros((numblockwidth,numblockheight))
    for i in range(0,numblockwidth):
        for j in range(0, numblockheight):
            pixel_i = i*block_size
            pixel_j = j*block_size
            avg = np.mean(img[pixel_i:pixel_i+block_size,pixel_j:pixel_j+block_size])
            newimg[i,j] = avg
            #img[pixel_i:pixel_i+block_size,pixel_j:pixel_j+block_size] = avg

    return newimg

save_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/sharp_training/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir) 


data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
train_data_filename = data_dir + 'images_shuffled/'
train_labels_filename = data_dir + 'groundtruth_shuffled/' 
pred_filename = (os.path.dirname(os.path.realpath(__file__)))+'/predictions_test/result/'
patch_size = (total_window_size,total_window_size)

num_images = 311#311


patches = []
for i in range(1, num_images+1):
    imageid = "satImage_%.3d" % i
    image_filename = train_labels_filename + imageid + ".png"
    if os.path.isfile(image_filename):
        print ('Loading ' + image_filename)
        img = mpimg.imread(image_filename)
        img = mean_img_per_patch(img,PATCH_SIZE)

        img_patches = img_crop_context(img, 1, 1,CTX_SIZE)
        img_patches = np.asarray(img_patches)
        shaped_data = np.reshape(img_patches,(img_patches.shape[0],-1)) 
        patches.extend(shaped_data) 
    else:
        print ('File ' + image_filename + ' does not exist')

print("Fetching some atoms")
patches = np.asarray(patches)
n_atoms = int(400/total_window_size)**2
print("Num atoms: "+str(n_atoms))
dico = MiniBatchDictionaryLearning(15)
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
    img = mean_img_per_patch(img,PATCH_SIZE)
    img_patches = img_crop_context(img,1,1,CTX_SIZE)
    img_patches = np.asarray(img_patches)

    new_arr = []
    for x in range(0,img_patches.shape[0]):
        dimx = img_patches[x].shape[0]
        dimy = img_patches[x].shape[1]
        if dimx*dimy == 5*5:
            new_arr.append(np.reshape(img_patches[x],(-1)))
        else:
            datav = np.pad(img_patches[x],((0,5-dimx),(0,5-dimy)),mode='reflect')
            datav = np.reshape(datav,(5*5))
            new_arr.append(datav)

    new_arr = np.asarray(new_arr)
    #shaped_data = np.reshape(img_patches,(1,80*80)) 
    U = sparse_encode(new_arr,V)
    reconstructed = np.dot(U,V)
    print(reconstructed)
    img_pred = label_to_img(38, 38, 1, 1, reconstructed)
    print(np.array_equal(img_pred,img))
    print(img_pred)
    print(img)
    f, (ax1, ax2) = plt.subplots(1, 2)
    img_pred  = np.rot90(img_pred,3)
    img_pred = np.fliplr(img_pred)
    img_pred = binarize(img_pred,1,0.3)
    ax1.imshow(img_pred,cmap='gray')
    ax1.set_title('Denoised')
    ax2.set_title('Original one')
    ax2.imshow(img,cmap='gray')
    plt.show()








