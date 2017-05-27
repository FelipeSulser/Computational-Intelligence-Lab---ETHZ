"""
Baseline for CIL project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss
"""
import gc
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
import scipy
from scipy import ndimage
import math 


NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2 

SEED = 66478  # Set to None for random seed.

BATCH_SIZE = 32 # has to be dividable by 2 so we can split 2 classes to be equal sized
#TODO change epoch number
NUM_EPOCHS = 200
RECORDING_STEP = 300
DOWNSCALE = 1

MODE = 'train' # 'train' or 'predict'
STARTING_ID = 1 # 21, 41...
TRAINING_SIZE = 50 #114

TEST_START_ID = 1 
TEST_SIZE = 50

VALIDATION_N_PATCHES = 0    # if 0, no validation is being done

init_type = 'xavier'

LOGGING = False
#CUDA_VISIBLE_DEVICES=""
# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!

CONTEXT_ADDITIVE_FACTOR = 24 #patch context increased by 2x2, so a 8x8 patch becomes a 16x15
IMG_PATCH_SIZE = 16 #should be at least dividor of 608
CONTEXT_PATCH = IMG_PATCH_SIZE+2*CONTEXT_ADDITIVE_FACTOR #in this case window is 16x16

FC1_WIDTH = 32768


tf.app.flags.DEFINE_string('train_dir', 'datafiles',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


def batch_norm(x, n_out, phase_train):
    print('phase_train: ', phase_train)
    if phase_train:
        phase_train = tf.constant(True)
    else:
        phase_train = tf.constant(False)
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed




# Extract patches from a given image
# the patch itself should be (w+2*context_factor,h+2*context_factor)
# but the base used for labeling should be (w,h)
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


def extract_data(filename, num_images, starting_id, context_factor):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(starting_id, num_images+starting_id):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)


            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = int(imgs[0].shape[0]/DOWNSCALE)
    IMG_HEIGHT = int(imgs[0].shape[1]/DOWNSCALE)
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)


    img_patches = [img_crop_context(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE,context_factor, sub_mean=True) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    data = np.asarray(data)
    return data
        
# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return [1, 0]
    else:
        return [0, 1]

# Extract label images
def extract_labels(filename, num_images, starting_id, context_factor):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(starting_id, num_images+starting_id):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            #img = Image.open(image_filename)
            #downscaled = img.resize((200,200)) #HARDCODED
            #downscaled = np.asarray(downscaled)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    # it means that we base our labels only on the core of the patch, not including the contet added
    context_factor = 0
    gt_patches = [img_crop_context(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE,context_factor) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = np.zeros([imgheight, imgwidth])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            # if labels[idx][0] > 0.5:
            #     l = 1
            # else:
            #     l = 0
            l = labels[idx][0]
            array_labels[i:i+h, j:j+w] = l

            idx = idx + 1
    return array_labels

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def main(argv=None):  # pylint: disable=unused-argument
    data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
    train_data_filename = data_dir + 'images_shuffled/'
    train_labels_filename = data_dir + 'groundtruth_shuffled/' 
    test_set_dir = (os.path.dirname(os.path.realpath(__file__)))+'/test_set_images/'
    # Extract it into numpy arrays.

    num_epochs = NUM_EPOCHS

    train_data = extract_data(train_data_filename, TRAINING_SIZE, STARTING_ID, CONTEXT_ADDITIVE_FACTOR)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE, STARTING_ID, CONTEXT_ADDITIVE_FACTOR)


    print("Train data shape: ", train_data.shape)
    print("Train labels shape: ", train_labels.shape)
    
    
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]

    class_road_size = len(idx0)
    class_back_size = len(idx1)

    print ('Number of data points per class: road = ' + str(class_road_size) + ' back = ' + str(class_back_size))
    min_c = min(class_road_size, class_back_size)
    print ('Balancing training data...')


    new_indices_all = idx0 + idx1

    np.random.shuffle(idx0)
    np.random.shuffle(idx1)

    new_indices = idx0[0:min_c] + idx1[0:min_c]
    
    #train_data_all = train_data[new_indices_all,:,:,:]
    train_data = train_data[new_indices,:,:,:]

    #train_labels_all = train_labels[new_indices_all]
    train_labels = train_labels[new_indices]

    train_size = 2 * min_c
    print('train_size: ', train_size)


    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))



    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, CONTEXT_PATCH, CONTEXT_PATCH, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))

    if VALIDATION_N_PATCHES > 0:
        validation_data_node = tf.placeholder(
            tf.float32,
            shape=(VALIDATION_N_PATCHES, CONTEXT_PATCH, CONTEXT_PATCH, NUM_CHANNELS))
        validation_labels_node = tf.placeholder(tf.float32,
                                           shape=(VALIDATION_N_PATCHES, NUM_LABELS))


    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}

    w_1 = 64   # with of the first chunk of layers
    w_2 = 128
    w_3 = 256
    w_4 = 512
    
    if init_type == 'xavier': 
        conv0_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv0_weights = tf.Variable( conv0_weights_init(shape=[7, 7, NUM_CHANNELS, 64]), name='conv0_weights')
        conv0_biases = tf.Variable(tf.constant(0.001, shape=[64]), name='conv0_biases')


        conv11_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv11_weights = tf.Variable( conv11_weights_init(shape=[3, 3, w_1, w_1]), name='conv11_weights')
        conv11_biases = tf.Variable(tf.constant(0.001, shape=[w_1]), name='conv11_biases')

        conv12_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv12_weights = tf.Variable( conv12_weights_init(shape=[3, 3, w_1, w_1]), name='conv12_weights')
        conv12_biases = tf.Variable(tf.constant(0.001, shape=[w_1]), name='conv12_biases')

        conv13_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv13_weights = tf.Variable( conv13_weights_init(shape=[3, 3, w_1, w_1]), name='conv13_weights')
        conv13_biases = tf.Variable(tf.constant(0.01, shape=[w_1]), name='conv13_biases')

        conv14_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv14_weights = tf.Variable( conv14_weights_init(shape=[3, 3, w_1, w_1]), name='conv14_weights')
        conv14_biases = tf.Variable(tf.constant(0.01,shape=[w_1]), name='conv14_biases')

        conv1_map_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv1_map_weights = tf.Variable( conv1_map_weights_init(shape=[1, 1, w_1, w_2]), name='conv1_map_weights')
        conv1_map_biases = tf.Variable(tf.constant(0.01,shape=[w_2]), name='conv1_map_biases')


        conv21_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv21_weights = tf.Variable( conv21_weights_init(shape=[3, 3, w_2, w_2]), name='conv21_weights')
        conv21_biases = tf.Variable(tf.constant(0.001, shape=[w_2]), name='conv21_biases')

        conv22_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv22_weights = tf.Variable( conv22_weights_init(shape=[3, 3, w_2, w_2]), name='conv22_weights')
        conv22_biases = tf.Variable(tf.constant(0.001, shape=[w_2]), name='conv22_biases')

        conv23_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv23_weights = tf.Variable( conv23_weights_init(shape=[3, 3, w_2, w_2]), name='conv23_weights')
        conv23_biases = tf.Variable(tf.constant(0.01, shape=[w_2]), name='conv23_biases')

        conv24_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv24_weights = tf.Variable( conv24_weights_init(shape=[3, 3, w_2, w_2]), name='conv24_weights')
        conv24_biases = tf.Variable(tf.constant(0.01,shape=[w_2]), name='conv24_biases')

        conv25_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv25_weights = tf.Variable( conv25_weights_init(shape=[3, 3, w_2, w_2]), name='conv25_weights')
        conv25_biases = tf.Variable(tf.constant(0.01, shape=[w_2]), name='conv25_biases')

        conv26_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv26_weights = tf.Variable( conv26_weights_init(shape=[3, 3, w_2, w_2]), name='conv26_weights')
        conv26_biases = tf.Variable(tf.constant(0.01,shape=[w_2]), name='conv26_biases')

        conv2_map_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv2_map_weights = tf.Variable( conv2_map_weights_init(shape=[1, 1, w_2, w_3]), name='conv2_map_weights')
        conv2_map_biases = tf.Variable(tf.constant(0.01,shape=[w_3]), name='conv2_map_biases')


        conv31_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv31_weights = tf.Variable( conv31_weights_init(shape=[3, 3, w_3, w_3]), name='conv31_weights')
        conv31_biases = tf.Variable(tf.constant(0.001, shape=[w_3]), name='conv31_biases')

        conv32_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv32_weights = tf.Variable( conv32_weights_init(shape=[3, 3, w_3, w_3]), name='conv32_weights')
        conv32_biases = tf.Variable(tf.constant(0.001, shape=[w_3]), name='conv32_biases')

        conv33_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv33_weights = tf.Variable( conv33_weights_init(shape=[3, 3, w_3, w_3]), name='conv33_weights')
        conv33_biases = tf.Variable(tf.constant(0.01, shape=[w_3]), name='conv33_biases')

        conv34_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv34_weights = tf.Variable( conv34_weights_init(shape=[3, 3, w_3, w_3]), name='conv34_weights')
        conv34_biases = tf.Variable(tf.constant(0.01,shape=[w_3]), name='conv34_biases')

        conv35_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv35_weights = tf.Variable( conv35_weights_init(shape=[3, 3, w_3, w_3]), name='conv35_weights')
        conv35_biases = tf.Variable(tf.constant(0.01, shape=[w_3]), name='conv35_biases')

        conv36_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv36_weights = tf.Variable( conv36_weights_init(shape=[3, 3, w_3, w_3]), name='conv36_weights')
        conv36_biases = tf.Variable(tf.constant(0.01,shape=[w_3]), name='conv36_biases')

        conv37_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv37_weights = tf.Variable( conv37_weights_init(shape=[3, 3, w_3, w_3]), name='conv37_weights')
        conv37_biases = tf.Variable(tf.constant(0.01, shape=[w_3]), name='conv37_biases')

        conv38_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv38_weights = tf.Variable( conv38_weights_init(shape=[3, 3, w_3, w_3]), name='conv38_weights')
        conv38_biases = tf.Variable(tf.constant(0.01,shape=[w_3]), name='conv38_biases')

        # conv39_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        # conv39_weights = tf.Variable( conv39_weights_init(shape=[3, 3, w_3, w_3]), name='conv39_weights')
        # conv39_biases = tf.Variable(tf.constant(0.01, shape=[w_3]), name='conv39_biases')

        # conv310_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        # conv310_weights = tf.Variable( conv310_weights_init(shape=[3, 3, w_3, w_3]), name='conv310_weights')
        # conv310_biases = tf.Variable(tf.constant(0.01,shape=[w_3]), name='conv310_biases')

        conv3_map_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv3_map_weights = tf.Variable( conv3_map_weights_init(shape=[1, 1, w_3, w_4]), name='conv3_map_weights')
        conv3_map_biases = tf.Variable(tf.constant(0.01,shape=[w_4]), name='conv3_map_biases')


        conv41_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv41_weights = tf.Variable( conv41_weights_init(shape=[3, 3, w_4, w_4]), name='conv41_weights')
        conv41_biases = tf.Variable(tf.constant(0.001, shape=[w_4]), name='conv41_biases')

        conv42_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv42_weights = tf.Variable( conv42_weights_init(shape=[3, 3, w_4, w_4]), name='conv42_weights')
        conv42_biases = tf.Variable(tf.constant(0.001, shape=[w_4]), name='conv42_biases')

        conv43_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv43_weights = tf.Variable( conv43_weights_init(shape=[3, 3, w_4, w_4]), name='conv43_weights')
        conv43_biases = tf.Variable(tf.constant(0.01, shape=[w_4]), name='conv43_biases')

        conv44_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv44_weights = tf.Variable( conv44_weights_init(shape=[3, 3, w_4, w_4]), name='conv44_weights')
        conv44_biases = tf.Variable(tf.constant(0.01, shape=[w_4]), name='conv44_biases')

        conv45_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv45_weights = tf.Variable( conv45_weights_init(shape=[3, 3, w_4, w_4]), name='conv45_weights')
        conv45_biases = tf.Variable(tf.constant(0.01, shape=[w_4]), name='conv45_biases')

        conv46_weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        conv46_weights = tf.Variable( conv46_weights_init(shape=[3, 3, w_4, w_4]), name='conv46_weights')
        conv46_biases = tf.Variable(tf.constant(0.01, shape=[w_4]), name='conv46_biases')



        fc1_weights_init = tf.contrib.layers.xavier_initializer(uniform=False)
        fc1_weights = tf.Variable(  fc1_weights_init(shape=[FC1_WIDTH, NUM_LABELS]), name='fc1_weights')
        fc1_biases = tf.Variable(tf.constant(0.01, shape=[NUM_LABELS]), name='fc1_biases')

    else: 
        print('You have to specify some init_type')
       

    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx = 0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value*PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V
    
        # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image 
    def get_prediction(img):
        cropped = img_crop_context(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE,CONTEXT_ADDITIVE_FACTOR, sub_mean=True)
        data = np.asarray(cropped)

        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

        return img_prediction
    
    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        data = tf.cast(data, dtype=tf.float32)
        # print(tf.shape(conv0_weights))
        conv0 = tf.nn.conv2d(data,
                            conv0_weights,
                            strides=[1, 2, 2, 1], 
                            padding='SAME')
        relu0 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv0, CONTEXT_PATCH, train), conv0_biases))

        pool1 = tf.nn.max_pool(relu0,
                                ksize=[1,2,2,1],
                                strides=[1,2,2,1],
                                padding='SAME')

        conv11 = tf.nn.conv2d(pool1,
                            conv11_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu11 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv11, 64, train), conv11_biases))
       
        conv12 = tf.nn.conv2d(relu11,
                            conv12_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu12 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv12, 64, train),conv12_biases))

        relu_add_1 = tf.nn.relu(tf.add(pool1, relu12))

        conv13 = tf.nn.conv2d(relu_add_1,
                            conv13_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu13 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv13, 64, train), conv13_biases))
       
        conv14 = tf.nn.conv2d(relu13,
                            conv14_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu14 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv14, 64, train),conv14_biases))
        
        relu_add_2_ = tf.nn.relu(tf.add(relu_add_1, relu14))
        conv1_map = tf.nn.conv2d(relu_add_2_,
                            conv1_map_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu_add_2 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv1_map, 128, train),conv1_map_biases))





        conv21 = tf.nn.conv2d(relu_add_2,
                            conv21_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu21 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv21, 128, train), conv21_biases))
       
        conv22 = tf.nn.conv2d(relu21,
                            conv22_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu22 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv22, 128, train),conv22_biases))

        relu_add_3 = tf.nn.relu(tf.add(relu_add_2, relu22))

        conv23 = tf.nn.conv2d(relu_add_3,
                            conv23_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu23 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv23, 128, train), conv23_biases))
       
        conv24 = tf.nn.conv2d(relu23,
                            conv24_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu24 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv24, 128, train),conv24_biases))
        
        relu_add_4 = tf.nn.relu(tf.add(relu_add_3, relu24))

        conv25 = tf.nn.conv2d(relu_add_4,
                            conv25_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu25 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv25, 128, train), conv25_biases))
       
        conv26 = tf.nn.conv2d(relu25,
                            conv26_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu26 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv26, 128, train),conv26_biases))
        
        relu_add_5_ = tf.nn.relu(tf.add(relu_add_4, relu26))
        conv2_map = tf.nn.conv2d(relu_add_5_,
                            conv2_map_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu_add_5 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv2_map, 256, train),conv2_map_biases))



        conv31 = tf.nn.conv2d(relu_add_5,
                            conv31_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu31 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv31, 256, train), conv31_biases))
       
        conv32 = tf.nn.conv2d(relu31,
                            conv32_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu32 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv32, 256, train),conv32_biases))

        relu_add_6 = tf.nn.relu(tf.add(relu_add_5, relu32))

        conv33 = tf.nn.conv2d(relu_add_6,
                            conv33_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu33 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv33, 256, train), conv33_biases))
       
        conv34 = tf.nn.conv2d(relu33,
                            conv34_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu34 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv34, 256, train),conv34_biases))
        
        relu_add_7 = tf.nn.relu(tf.add(relu_add_6, relu34))

        conv35 = tf.nn.conv2d(relu_add_7,
                            conv35_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu35 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv35, 256, train), conv35_biases))
       
        conv36 = tf.nn.conv2d(relu35,
                            conv36_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu36 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv36, 256, train),conv36_biases))
        
        relu_add_8 = tf.nn.relu(tf.add(relu_add_7, relu36))

        conv37 = tf.nn.conv2d(relu_add_8,
                            conv37_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu37 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv37, 256, train), conv37_biases))
       
        conv38 = tf.nn.conv2d(relu37,
                            conv38_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu38 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv38, 256, train),conv38_biases))
        
        relu_add_9_ = tf.nn.relu(tf.add(relu_add_8, relu38))
        conv3_map = tf.nn.conv2d(relu_add_9_,
                            conv3_map_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu_add_9 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv3_map, 512, train),conv3_map_biases))



        conv41 = tf.nn.conv2d(relu_add_9,
                            conv41_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu41 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv41, 512, train), conv41_biases))
       
        conv42 = tf.nn.conv2d(relu41,
                            conv42_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu42 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv42, 512, train),conv42_biases))

        relu_add_10 = tf.nn.relu(tf.add(relu_add_9, relu42))

        conv43 = tf.nn.conv2d(relu_add_10,
                            conv43_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu43 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv43, 512, train), conv43_biases))
       
        conv44 = tf.nn.conv2d(relu43,
                            conv44_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu44 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv44, 512, train),conv44_biases))
        
        relu_add_11 = tf.nn.relu(tf.add(relu_add_10, relu44))

        conv45 = tf.nn.conv2d(relu_add_11,
                            conv45_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu45 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv45, 512, train), conv45_biases))
       
        conv46 = tf.nn.conv2d(relu45,
                            conv46_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu46 = tf.nn.relu(tf.nn.bias_add(batch_norm(conv46, 512, train),conv46_biases))
        
        relu_add_12 = tf.nn.relu(tf.add(relu_add_11, relu46))



        pool2 = tf.nn.avg_pool(relu_add_12,
                                ksize=[1,2,2,1],
                                strides=[1,2,2,1],
                                padding='SAME')
        

        # if train:
        #     print("relu1: "+str(relu1.get_shape()))
        #     print("Pool1: "+str(pool1.get_shape()))
        #     print("relu2: "+str(relu2.get_shape()))
        #     print("pool2: "+str(pool2.get_shape()))
        #     print("relu3: "+str(relu3.get_shape()))
        #     print("pool3: "+str(pool3.get_shape()))
        #     print("relu4: "+str(relu4.get_shape()))
        #     print("pool4: "+str(pool4.get_shape()))

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        print("pool1: "+str(pool2.get_shape()))
        print("pool2: "+str(pool2.get_shape()))
        relu_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [relu_shape[0], relu_shape[1] * relu_shape[2] * relu_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        
        
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        # if train:
        #     hidden = tf.nn.dropout(hidden1, 0.5, seed=SEED)
        #     hidden2 = tf.nn.dropout(hidden2,0.5,seed=SEED)

        out = tf.matmul(reshape, fc1_weights) + fc1_biases


        if train == True and LOGGING:
            summary_id = '_0'
            s_data = get_image_summary(data)
            filter_summary0 = tf.summary.image('summary_data' + summary_id, s_data)
            s_conv = get_image_summary(conv2)
            filter_summary2 = tf.summary.image('summary_conv' + summary_id, s_conv)
            s_pool1 = get_image_summary(pool1)
            filter_summary3 = tf.summary.image('summary_pool' + summary_id, s_pool1)
            s_conv2 = get_image_summary(conv2)
            filter_summary4 = tf.summary.image('summary_conv2' + summary_id, s_conv2)
            s_pool2 = get_image_summary(pool2)
            filter_summary5 = tf.summary.image('summary_pool' + summary_id, s_pool2)
            s_conv3 = get_image_summary(conv3)
            filter_summary6 = tf.summary.image('summary_conv3'+summary_id,s_conv3)
            s_pool3 = get_image_summary(pool1)
            filter_summary7 = tf.summary.image('summary_pool' + summary_id, s_pool3)
            s_conv4 = get_image_summary(conv4)
            filter_summary8 = tf.summary.image('summary_conv4'+summary_id,s_conv4)

        return out

    # Training computation: logits + cross-entropy loss.
    if MODE=='train':
        tr = True
    else:
        tr = False
    logits = model(train_data_node, tr) # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=train_labels_node))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0, name='batch')
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        1.00,                # Decay rate.
        staircase=True)
    #learning_rate = 0.01
    tf.summary.scalar('learning_rate', learning_rate)
    
    #Momentum
    #optimizer = tf.train.MomentumOptimizer(learning_rate,
    #                                       momentum=0.2).minimize(loss,
    #                                                     global_step=batch)
    
    # AdamOptimizer - adaptative momentum
    # learning_rate: 1e-4 is very often used. ADAM chooses itsself a learning rate, 
    #               so tf.train.exponential_decay might not be a good idea
    # epsilon: 0.1 as recommended for imagenet
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, 
            beta1= 0.9, beta2 = 0.999,epsilon=0.1).minimize(loss,global_step=batch)


    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Number of variables in model: "+str(total_parameters))




    # FOR VALIDATION
    if VALIDATION_N_PATCHES > 0:
        validation_logits = model(validation_data_node) 
        validation_pred = tf.nn.softmax(model(validation_data_node))
        validation_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=validation_logits, labels=validation_labels_node))
        validation_loss += 5e-4 * regularizers





    img_prediction = None
    #tf.get_default_graph().finalize()
    # Create a local session to run this computation.
    with tf.Session() as s:

        if MODE == 'predict':
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")


            #plotNNFilter(conv1_weights)
            print ("Running prediction on training set")
            prediction_training_dir = "predictions_training/"
            real_prediction = "predictions_test/"
            if not os.path.isdir(prediction_training_dir):
                os.mkdir(prediction_training_dir)
            if not os.path.isdir(real_prediction):
                os.mkdir(real_prediction)
            if not os.path.isdir(real_prediction+"result/"):
                os.mkdir(real_prediction+"result/")


            n_patches_in_test_img = int(math.ceil(608 / IMG_PATCH_SIZE) ** 2)
            print('n_patches_in_test_img: ', n_patches_in_test_img)
            predict_data_node = tf.placeholder( tf.float32,
                    shape=(n_patches_in_test_img, CONTEXT_PATCH, CONTEXT_PATCH, NUM_CHANNELS))


            for i in range(TEST_START_ID, TEST_START_ID+TEST_SIZE):
                print("Prediction for img: "+str(i))

                # read image
                fname = "test_"+str(i)
                image_filename = test_set_dir + fname+"/"+fname + ".png"
                img = mpimg.imread(image_filename)

                # predict label
                cropped = img_crop_context(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE,CONTEXT_ADDITIVE_FACTOR,sub_mean=True)
                cropped = np.asarray(cropped)


                output = tf.nn.softmax(model(predict_data_node))
                output_prediction = s.run(output, feed_dict={predict_data_node: cropped})
                img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)




                # overlay and save
                oimg = make_img_overlay(img, img_prediction)
                oimg.save(real_prediction + "overlay_" + str(i) + ".png")
                oimg.close()
                # save the mask separately
                
                #need to multiply by 255 so its a real white pixel
                imgdata = Image.fromarray(255*img_prediction)
                imgdata = imgdata.convert('RGB')
                imgdata.save(real_prediction+"result/"+"prediction_"+str(i)+".png")
                imgdata.close()
                gc.collect()

        elif MODE == 'train':

            if STARTING_ID == 1:
                # we have no model yet, so lets init all variables
                s.run(init_op)
                print ('Initialized!')
            else:
                # we have the model already saved, so we dont need to init, but restore the existing model
                saver.restore(s, FLAGS.train_dir + "/model.ckpt")


            if LOGGING:
                # Build the summary operation based on the TF collection of Summaries.
                summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                                        graph=s.graph)
            
            # Loop through training steps.

            print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))



            for iepoch in range(num_epochs):


                # VALIDATION SET
                validation_idx = np.random.permutation(train_size)[:VALIDATION_N_PATCHES]



                # Permute training indices
                perm_indices = np.random.permutation(train_size)

                for step in range (int(train_size / BATCH_SIZE)):
                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]
                    

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]

                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    if step % RECORDING_STEP == 0:

                        if LOGGING:
                            summary_str, _, l, lr, predictions = s.run(
                            [summary_op, optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
                            summary_writer.add_summary(summary_str, step)
                            summary_writer.flush()
                        else:
                            if VALIDATION_N_PATCHES > 0:
                                feed_dict[validation_data_node] = train_data[validation_idx, :, :, :]
                                feed_dict[validation_labels_node] = train_labels[validation_idx]
                                validation_labels = train_labels[validation_idx]

                                _, l, lr, predictions, validation_out, valid_loss = s.run(
                                    [ optimizer, loss, learning_rate, train_prediction, 
                                    validation_pred, validation_loss],
                                    feed_dict=feed_dict)
                            else:
                                _, l, lr, predictions = s.run(
                                    [ optimizer, loss, learning_rate, train_prediction],
                                    feed_dict=feed_dict)


                        print ('Epoch %.2f' % (iepoch))
                        print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))
                        if VALIDATION_N_PATCHES > 0:
                            print ('Validation loss: %.3f' % (valid_loss))
                            print ('Validation error: %.1f%%' % error_rate(validation_out,
                                                                         validation_labels))

                        sys.stdout.flush()
                    else:
                        
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)




                # Save the variables to disk.
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)



         


if __name__ == '__main__':

    tf.app.run()