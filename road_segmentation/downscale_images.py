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

    train_data_filename_out = data_dir+'images_downscaled/'
    train_labels_filename_out = data_dir+'groundtruth_downscaled/'

    FACTOR = 0.5

    if not os.path.isdir(train_data_filename_out):
        os.mkdir(train_data_filename_out)

    if not os.path.isdir(train_labels_filename_out):
        os.mkdir(train_labels_filename_out)


    for i in range(1, 201):
        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            downscaled = img.resize((200,200)) #HARDCODED
            downscaled.save(train_data_filename_out+imageid+".png")
        

        label_image_filename = train_labels_filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + label_image_filename)
            img = Image.open(label_image_filename)
            downscaled = img.resize((200,200)).convert("L") #HARDCODED
            data_array = numpy.asarray(downscaled)
            
            mydata = data_array
            mydata.flags.writeable = True
            mydata[mydata > 128] = 255
            mydata[mydata<= 128 ] = 0
           
            img_bw = Image.fromarray(mydata,'L')
            img_bw.save(train_labels_filename_out+imageid+".png")
            #downscaled.save(train_labels_filename_out+imageid+".png")
            



        else:
            print ('File ' + image_filename + ' does not exist')
