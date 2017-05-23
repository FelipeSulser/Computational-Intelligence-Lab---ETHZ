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

DOUBLE_AUGMENT = [23,26,27,42,72,83,91]


if __name__ == '__main__':
    imgs = []
    data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/' 
    out_data_dir = data_dir+'images_extended/'
    if not os.path.isdir(out_data_dir):
        os.mkdir(out_data_dir) 

    out_label_dir = data_dir+'groundtruth_extended/'
    if not os.path.isdir(out_label_dir):
        os.mkdir(out_label_dir) 
    for i in range(1, 101):

        #Transpose rotation

        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            img2 = img.rotate(90)
            new_ix = 100+i
            new_imageid = "satImage_%.3d" % new_ix
            img2.save(out_data_dir+new_imageid+".png")

            #save the original image too
            img.save(out_data_dir+"satImage_"+str(i)+".png")
           

        label_image_filename = train_labels_filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + label_image_filename)
            labelimg = Image.open(label_image_filename)
            labelimg2 = labelimg.rotate(90)
            label_new_ix = 100+i
            label_new_imageid = "satImage_%.3d" % label_new_ix
            labelimg2.save(out_label_dir+label_new_imageid+".png")
            #save original label too
            labelimg.save(out_label_dir+"satImage_"+str(i)+".png")

        #Mirror rotation

        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            img2 = img.rotate(180)
            new_ix = 200+i
            new_imageid = "satImage_%.3d" % new_ix
            img2.save(out_data_dir+new_imageid+".png")
           

        label_image_filename = train_labels_filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + label_image_filename)
            labelimg = Image.open(label_image_filename)
            labelimg2 = labelimg.rotate(180)
            label_new_ix = 200+i
            label_new_imageid = "satImage_%.3d" % label_new_ix
            labelimg2.save(out_label_dir+label_new_imageid+".png")
            
        else:
            print ('File ' + image_filename + ' does not exist')

    save_start_index = 1

    NEW_IDX_START = 300
    ROTATION = 180

    # Handpicked diagonal roads to augment dataset even further
    for i in DOUBLE_AUGMENT:
        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        label_image_filename = train_labels_filename + imageid + ".png"

        if os.path.isfile(image_filename) and os.path.isfile(label_image_filename):
            img = Image.open(image_filename)
            img2 = img.rotate(ROTATION)
            new_ix = NEW_IDX_START + save_start_index
            new_imageid = "satImage_%.3d" % new_ix
            img2.save(out_data_dir+new_imageid+".png")
           
            print ('Loading ' + label_image_filename)
            labelimg = Image.open(label_image_filename)
            labelimg2 = labelimg.rotate(ROTATION)
            label_new_ix = NEW_IDX_START + save_start_index
            label_new_imageid = "satImage_%.3d" % label_new_ix
            labelimg2.save(out_label_dir+label_new_imageid+".png")

            save_start_index += 1   

        else:
            print ('File ' + image_filename + ' does not exist')

    NEW_IDX_START = 307
    ROTATION = 270
    save_start_index = 1

    for i in DOUBLE_AUGMENT:
        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        label_image_filename = train_labels_filename + imageid + ".png"

        if os.path.isfile(image_filename) and os.path.isfile(label_image_filename):
            img = Image.open(image_filename)
            img2 = img.rotate(ROTATION)
            new_ix = NEW_IDX_START + save_start_index
            new_imageid = "satImage_%.3d" % new_ix
            img2.save(out_data_dir+new_imageid+".png")
           
            print ('Loading ' + label_image_filename)
            labelimg = Image.open(label_image_filename)
            labelimg2 = labelimg.rotate(ROTATION)
            label_new_ix = NEW_IDX_START + save_start_index
            label_new_imageid = "satImage_%.3d" % label_new_ix
            labelimg2.save(out_label_dir+label_new_imageid+".png")

            save_start_index += 1   

        else:
            print ('File ' + image_filename + ' does not exist')