import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import code
import tensorflow.python.platform
import numpy
import tensorflow as tf
from scipy import ndimage
import math

if __name__ == '__main__':
    imgs = []
    data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/' 
    for i in range(1, 101):
        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            img2 = img.rotate(90)
            new_ix = 100+i
            new_imageid = "satImage_%.3d" % new_ix
            img2.save(train_data_filename+new_imageid+".png")
           

        label_image_filename = train_labels_filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + label_image_filename)
            labelimg = Image.open(label_image_filename)
            labelimg2 = labelimg.rotate(90)
            label_new_ix = 100+i
            label_new_imageid = "satImage_%.3d" % label_new_ix
            labelimg2.save(train_labels_filename+label_new_imageid+".png")
            



        else:
            print ('File ' + image_filename + ' does not exist')
