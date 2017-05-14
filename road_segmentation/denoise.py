
from time import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import scipy.misc
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version
from skimage import data, img_as_float

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def remove_filtering_neighbors(img, num_neigh,threshold=3, block_size = 16):
    #img is b&w array with 0 or 1
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    
    numblockwidth = int(imgwidth/block_size)

    numblockheight = int(imgheight/block_size)

    for i in range(1,numblockwidth-1):
        for j in range(1, numblockheight-1):
            pixel_i = i*block_size
            pixel_j = j*block_size

            if img[pixel_i,pixel_j] == 0: #if patch is black
                #if not surrounded by 3 cut it
                neighbors = np.zeros(4)

                #black is 0

                neighbors[0] = img[pixel_i-block_size,pixel_j]
                neighbors[1] = img[pixel_i+block_size,pixel_j]
                neighbors[2] = img[pixel_i,pixel_j-block_size]
                neighbors[3] = img[pixel_i,pixel_j+block_size]

                sum_val = np.sum(neighbors)
                if(sum_val > threshold):
                    #repaint block
                    for xx in range(0,block_size):
                        for yy in range(0,block_size):
                            img[pixel_i+xx,pixel_j+yy] = 1.0
            else: #white patch 1
                #if not surrounded by 3 cut it
                neighbors = np.zeros(4)

                #black is 0

                neighbors[0] = img[pixel_i-block_size,pixel_j]
                neighbors[1] = img[pixel_i+block_size,pixel_j]
                neighbors[2] = img[pixel_i,pixel_j-block_size]
                neighbors[3] = img[pixel_i,pixel_j+block_size]

                sum_val = np.sum(neighbors)
                if(sum_val < 1):
                    #repaint block
                    for xx in range(0,block_size):
                        for yy in range(0,block_size):
                            img[pixel_i+xx,pixel_j+yy] = 0.0

    return img
save_dir = "/Users/felipesulser/Dropbox/ETH/CIL/TensorFlow/felipe_prediction/result_denoised/"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir) 
for i in range(1, 51):
    mydir = "/Users/felipesulser/Dropbox/ETH/CIL/TensorFlow/felipe_prediction/result/"
    imageid = "prediction_"+str(i)
    image_filename = mydir +imageid+ ".png"
    img = mpimg.imread(image_filename)
    img = rgb2gray(img)

    img_denoised = remove_filtering_neighbors(img,4)
    save_str = save_dir+imageid+".png"
    scipy.misc.imsave(save_str,img_denoised)







