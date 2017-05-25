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
import math

DOUBLE_AUGMENT = [23,26,27,42,72,83,91]
TO_REMOVE = [33]
MIRROR_ALL = True


if __name__ == '__main__':
    imgs = []
    data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/' 

    out_data_dir = data_dir+'images_extended/'
    out_label_dir = data_dir+'groundtruth_extended/'
    if not os.path.isdir(out_data_dir):
        os.mkdir(out_data_dir) 
    if not os.path.isdir(out_label_dir):
        os.mkdir(out_label_dir) 

    j = 1
    for i in range(1, 101):

        if i in TO_REMOVE:
            continue

        # Transpose rotation
        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        label_image_filename = train_labels_filename + imageid + ".png"
        if os.path.isfile(image_filename) and os.path.isfile(label_image_filename):
            img = Image.open(image_filename)
            img2 = img.rotate(90)
            new_ix = 100+j
            new_imageid = "satImage_%.3d" % new_ix
            img2.save(out_data_dir+new_imageid+".png")

            #save the original image too
            imageid = "satImage_%.3d" % j
            img.save(out_data_dir+imageid+".png")
           

            print ('Loading ' + label_image_filename)
            labelimg = Image.open(label_image_filename)
            labelimg2 = labelimg.rotate(90)

            labelimg2.save(out_label_dir+new_imageid+".png")
            #save original label too
            labelimg.save(out_label_dir+imageid+".png")


            #Mirror rotation
            if MIRROR_ALL:
                img2 = np.fliplr(img)
                new_ix_mirror = 200+j
                new_imageid = "satImage_%.3d" % new_ix_mirror
                img2 = Image.fromarray(img2)
                img2.save(out_data_dir+new_imageid+".png")

                labelimg2 = np.fliplr(labelimg)

                # TODO: felipe chech this :D
                labelimg2 = Image.fromarray(labelimg2)
                labelimg2.save(out_label_dir+new_imageid+".png")

            j += 1
            
        else:
            print ('File ' + image_filename + 'or its label file does not exist')

    save_start_index = 1

    if MIRROR_ALL:
        NEW_IDX_START = 300 - len(TO_REMOVE)
    else:
        NEW_IDX_START = 200 - len(TO_REMOVE)
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

    NEW_IDX_START = NEW_IDX_START + len(DOUBLE_AUGMENT)
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