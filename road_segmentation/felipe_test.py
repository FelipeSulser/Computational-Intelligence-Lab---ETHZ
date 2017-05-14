
from __future__ import division

import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np

from tf_image_segmentation.models.fcn_16s import FCN_16s
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)

sys.path.append("/Users/felipesulser/Dropbox/ETH/CIL/TensorFlow/tf-image-segmentation/")
sys.path.append("/Users/felipesulser/Dropbox/ETH/CIL/TensorFlow/models/slim/")

fcn_16s_checkpoint_path = \
 '/Users/felipesulser/Dropbox/ETH/CIL/TensorFlow/tf-image-segmentation/tf_image_segmentation/fcn_16s_checkpoint/model_fcn16s_final.ckpt'

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #'1'

slim = tf.contrib.slim
TRAINING_SIZE = 100
number_of_classes = 2
TRAIN_MODE = True
num_epochs = 8

image_filename = '/Users/felipesulser/Dropbox/ETH/CIL/TensorFlow/road_segmentation/sat.jpg'

data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/' 
predict_dir = (os.path.dirname(os.path.realpath(__file__)))+'/test_set_images/'
#image_filename = 'small_cat.jpg'
NUMFILES = len([name for name in os.listdir(predict_dir) if name != ".DS_Store"])
NAMEFILES = [name for name in os.listdir(predict_dir) if name != ".DS_Store"]
image_filename_placeholder = tf.placeholder(tf.string)

train_data = extract_data(train_data_filename,TRAINING_SIZE)
train_labels = extract_labels(train_labels_filename,TRAINING_SIZE)

c0 = 0
c1 = 0
for i in range(len(train_labels)):
    if train_labels[i][0] == 1:
        c0 = c0 + 1
    else:
        c1 = c1 + 1
print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

print ('Balancing training data...')
min_c = min(c0, c1)
idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
new_indices = idx0[0:min_c] + idx1[0:min_c]
print (len(new_indices))
print (train_data.shape)
train_data = train_data[new_indices,:,:,:]
train_labels = train_labels[new_indices]
train_size = train_labels.shape[0]

 c0 = 0
c1 = 0
for i in range(len(train_labels)):
    if train_labels[i][0] == 1:
        c0 = c0 + 1
    else:
        c1 = c1 + 1
print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

feed_dict_to_use = {image_filename_placeholder: image_filename}

image_tensor = tf.read_file(image_filename_placeholder)

image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)

# Fake batch for image and annotation by adding
# leading empty axis.
image_batch_tensor = tf.expand_dims(image_tensor, axis=0)

# Be careful: after adaptation, network returns final labels
# and not logits
FCN_16s = adapt_network_for_any_size_input(FCN_16s, 32)


pred, fcn_16s_variables_mapping = FCN_16s(image_batch_tensor=image_batch_tensor,
                                          number_of_classes=number_of_classes,
                                          is_training=TRAIN_MODE)

# The op for initializing the variables.
initializer = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    if TRAIN_MODE:
        sess.run(initializer)
        #tf.initialize_all_variables.run()
        training_indices = range(train_size)
        for iepoch in range(num_epochs):




    else:
        #iterate over the prediction dataset and generate output
        sess.run(initializer)

        saver.restore(sess,
         "/Users/felipesulser/Dropbox/ETH/CIL/TensorFlow/tf-image-segmentation/tf_image_segmentation/fcn_16s_checkpoint/model_fcn16s_final.ckpt")
        
        image_np, pred_np = sess.run([image_tensor, pred], feed_dict=feed_dict_to_use)
        
        io.imshow(image_np)
        io.show()
        
        io.imshow(pred_np.squeeze())
        io.show()