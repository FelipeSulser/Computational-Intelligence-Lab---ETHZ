import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import code
import tensorflow.python.platform
import numpy as np
import tensorflow as tf
from scipy import ndimage
import scipy
import math


if __name__ == '__main__':
    imgs = []
    data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
    train_data_filename = data_dir + 'images/'
    train_groundtruth_filename = data_dir + 'groundtruth/' 

    custom_dir = (os.path.dirname(os.path.realpath(__file__)))+'/custom_test/'
    out_data_dir = custom_dir +'images/'
    out_truth_dir = custom_dir +'groundtruth/'
    if not os.path.isdir(out_data_dir):
        os.mkdir(out_data_dir) 
    if not os.path.isdir(out_truth_dir):
        os.mkdir(out_truth_dir) 

    new_imageid = 1
    for i in range(1, 101):
        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        groundtruth_image_filename = train_groundtruth_filename + imageid + ".png"
        if os.path.isfile(image_filename) and os.path.isfile(groundtruth_image_filename):
            img_data = Image.open(image_filename)
            img_truth = Image.open(groundtruth_image_filename)

            # rotate 270 and mirror both directions
            img_data_raw = np.flipud(img_data.rotate(270))
            img_truth_raw = np.flipud(img_truth.rotate(270))

            # expand to final size
            N = 608 - 400
            img_data_raw = np.concatenate((img_data_raw, img_data_raw[-N-1:-1][::-1]), axis=0)
            img_data_raw = np.concatenate((img_data_raw, np.fliplr(img_data_raw[:,-N-1:-1])), axis=1)
            img_data = Image.fromarray(img_data_raw)
            
            img_truth_raw = np.concatenate((img_truth_raw, img_truth_raw[-N-1:-1][::-1]), axis=0)
            img_truth_raw = np.concatenate((img_truth_raw, np.fliplr(img_truth_raw[:,-N-1:-1])), axis=1)
            img_truth = Image.fromarray(img_truth_raw)
            
            # save files
            print(image_filename, 'parsed')
            img_data.save(out_data_dir + 'img_' + str(new_imageid) + ".png")
            img_truth.save(out_truth_dir + 'img_' + str(new_imageid) + ".png") 
            new_imageid += 1

        else:
            print ('File ' + image_filename + 'or its label file does not exist')