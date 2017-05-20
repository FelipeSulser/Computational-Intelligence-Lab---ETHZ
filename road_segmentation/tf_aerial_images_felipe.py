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
import numpy
import tensorflow as tf
from scipy import ndimage
import math 

NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2

VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
#TODO change batch size
BATCH_SIZE = 32 # 64
#TODO change epoch number
NUM_EPOCHS = 9
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
DOWNSCALE = 1

MODE = 'predict' # 'train' or 'predict'
STARTING_ID = 1 # 21, 41...
TRAINING_SIZE = 10



TEST_START_ID = 1
TEST_SIZE = 3





# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
#TODO change patch size

CONTEXT_ADDITIVE_FACTOR = 19 #patch context increased by 2x2, so a 8x8 patch becomes a 16x15
IMG_PATCH_SIZE = 12 #should be at least dividor of 608
CONTEXT_PATCH = IMG_PATCH_SIZE+2*CONTEXT_ADDITIVE_FACTOR #in this case window is 16x16


if CONTEXT_PATCH == 40:
    FC1_WIDTH = 576
elif CONTEXT_PATCH == 64:
    FC1_WIDTH = 1024
elif CONTEXT_PATCH == 32:
    FC1_WIDTH = 512
elif CONTEXT_PATCH == 50:
    FC1_WIDTH = 1024    
else:
    FC1_WIDTH = 42 # TODO 
    print('Please set FC1_WIDTH!!')


NUMFILES = 0
NAMEFILES = []



tf.app.flags.DEFINE_string('train_dir', 'datafiles',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS

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


# Extract patches from a given image
# the patch itself should be (w+2*context_factor,h+2*context_factor)
# but the base used for labeling should be (w,h)
def img_crop_context(im, w, h,context_factor):

    padding_type = 'reflect'
    cf = context_factor
    is_2d = len(im.shape) < 3
    if is_2d:
        padded_img = numpy.pad(im, cf, padding_type)
    else:
        padded_img = numpy.pad(im, ((cf,cf),(cf,cf),(0,0)), padding_type)




    list_patches = []
    imgheight = padded_img.shape[0]
    imgwidth = padded_img.shape[1]
    
    #print('padded_img: ', padded_img.shape)


    for i in range(cf,imgheight-cf,h):
        for j in range(cf,imgwidth-cf,w):
            im_patch = numpy.zeros(1)
            
            if is_2d:
                im_patch = padded_img[i-cf:i+h+cf, j-cf:j+w+cf]
                if im_patch.shape[0] < 2*cf+h and im_patch.shape[1] == 2*cf+w:
                    pad_size = 2*cf+h - im_patch.shape[0]
                    im_patch = numpy.pad(im_patch, ((0,pad_size),(0,0) ), padding_type)
                elif im_patch.shape[1] < 2*cf+w and im_patch.shape[0] == 2*cf+h:
                    pad_size = 2*cf+w - im_patch.shape[1]
                    im_patch = numpy.pad(im_patch, ((0,0),(0,pad_size)), padding_type)

                elif im_patch.shape[1] < 2*cf+w and im_patch.shape[0] < 2*cf+h:
                    pad_size0 = 2*cf+h - im_patch.shape[0]
                    pad_size1 = 2*cf+w - im_patch.shape[1]
                    im_patch = numpy.pad(im_patch, (( 0,pad_size0),(0,pad_size1)), padding_type)

            else:
                im_patch = padded_img[i-cf:i+h+cf, j-cf:j+w+cf, :]
                if im_patch.shape[0] < 2*cf+h and im_patch.shape[1] == 2*cf+w:
                    pad_size = 2*cf+h - im_patch.shape[0]
                    im_patch = numpy.pad(im_patch, ((0,pad_size),(0,0) ,(0,0)), padding_type)

                elif im_patch.shape[1] < 2*cf+w and im_patch.shape[0] == 2*cf+h:
                    pad_size = 2*cf+w - im_patch.shape[1]
                    im_patch = numpy.pad(im_patch, ((0,0),(0,pad_size), (0,0)), padding_type)

                elif im_patch.shape[1] < 2*cf+w and im_patch.shape[0] < 2*cf+h:
                    pad_size0 = 2*cf+h - im_patch.shape[0]
                    pad_size1 = 2*cf+w - im_patch.shape[1]
                    im_patch = numpy.pad(im_patch, (( 0,pad_size0),(0,pad_size1),(0,0)), padding_type)



            list_patches.append(im_patch)

    return list_patches

# OLD VERSION!!! use img_crop_context() instead
# def _img_crop_context_depr(im, w, h,context_factor):
#     list_patches = []
#     imgwidth = im.shape[0]
#     imgheight = im.shape[1]
#     is_2d = len(im.shape) < 3
#     for i in range(0,imgheight,h):
#         for j in range(0,imgwidth,w):
#             if is_2d:
#                 #im_patch = im[j:j+w, i:i+h]
#                 im_patch = numpy.zeros((w+2*context_factor,h+2*context_factor))
#                 iterx = 0
#                 itery = 0
#                 for x in range(j - context_factor,j+w+context_factor):
#                     itery = 0
#                     for y in range(i - context_factor,i+h+context_factor):
#                         if x >= 0 and y >= 0 and x < imgwidth and y < imgheight:
#                             im_patch[iterx,itery] = im[x,y]
                        
#                         itery = itery + 1
#                     iterx = iterx + 1
#                 #print("INDEX: ["+str(j-context_factor)+":"+str(j+w+context_factor)+", "+str(i-context_factor)+":"+str(i+h+context_factor)+"]")
#                 #im_patch = im[(j-context_factor):(j+w+context_factor), (i-context_factor):(i+h+context_factor)]
#             else:
#                 #im_patch = im[j:j+w, i:i+h, :]
#                 #print("INDEX: ["+str(j-context_factor)+":"+str(j+w+context_factor)+", "+str(i-context_factor)+":"+str(i+h+context_factor)+"]")
#                 #im_patch = im[(j-context_factor):(j+w+context_factor), (i-context_factor):(i+h+context_factor),:]
#                 im_patch = numpy.zeros((w+2*context_factor,h+2*context_factor,3))
#                 #print(im_patch.shape)
#                 iterx = 0
#                 itery = 0
#                 for x in range(j - context_factor,j+w+context_factor):
#                     itery = 0
#                     for y in range(i - context_factor,i+h+context_factor):
#                         #print(str(x) + " "+str(y)+" and "+str(iterx)+" " + str(itery))
#                         if x >= 0 and y >= 0 and x < imgwidth and y < imgheight:
#                             im_patch[iterx,itery,:] = im[x,y,:]
                        
#                         itery = itery + 1
#                     iterx = iterx + 1
#                 # img_data = Image.fromarray(numpy.uint8(im_patch*255))
#                 # plt.imshow(img_data)
#                 # plt.show()
#             #print(im_patch)
#             list_patches.append(im_patch)
#     #print(list_patches[250][:,:,0])
#     print('list_patches[0].shape: ', list_patches[0].shape)
#     return list_patches

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
            #img = Image.open(image_filename)
            #downscaled = img.resize((200,200)) #HARDCODED
            #downscaled = numpy.asarray(downscaled)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = int(imgs[0].shape[0]/DOWNSCALE)
    IMG_HEIGHT = int(imgs[0].shape[1]/DOWNSCALE)
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)


    img_patches = [img_crop_context(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE,context_factor) for i in range(num_images)]
    print('img_patches: ', len(img_patches))
    print('img_patches[0]: ', len(img_patches[0]))
    print('img_patches[0][0]: ', img_patches[0][0].shape)
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    print('data[0]: ', data[0].shape)
    data = numpy.asarray(data)
    return data
        
# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
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
            #downscaled = numpy.asarray(downscaled)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    # TODO: check with felipe if this is meaningful.
    #       it means that we base our labels only on the core of the patch, not including the contet added
    context_factor = 0
    gt_patches = [img_crop_context(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE,context_factor) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgheight, imgwidth])
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
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def main(argv=None):  # pylint: disable=unused-argument

    data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/' 
    test_set_dir = (os.path.dirname(os.path.realpath(__file__)))+'/test_set_images/'
    # Extract it into numpy arrays.

    NUMFILES = len([name for name in os.listdir(test_set_dir) if name != ".DS_Store"])
    NAMEFILES = [name for name in os.listdir(test_set_dir) if name != ".DS_Store"]

    num_epochs = NUM_EPOCHS



    train_data = extract_data(train_data_filename, TRAINING_SIZE, STARTING_ID, CONTEXT_ADDITIVE_FACTOR)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE, STARTING_ID, CONTEXT_ADDITIVE_FACTOR)

    print(type(train_data))
    print(train_data.shape)
    print(train_data)
    print(type(train_labels))
    print(train_labels.shape)
    print(train_labels)
    print("Train data shape: ", train_data.shape)
    print("Train labels shape: ", train_labels.shape)
    
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


    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, CONTEXT_PATCH, CONTEXT_PATCH, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))


    train_all_data_node = tf.constant(train_data)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}

    init_type = 'xavier'
    if init_type=='normal':
        conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, NUM_CHANNELS, 16],  # 5x5 filter, depth 32.
                                stddev=0.1,
                                seed=SEED), name='conv1_weights')
        conv1_biases = tf.Variable(tf.zeros([16]), name='conv1_biases')

        conv2_weights = tf.Variable(
            tf.truncated_normal([3, 3, 16, 32],
                                stddev=0.1,
                                seed=SEED), name='conv2_weights')
        conv2_biases = tf.Variable(tf.zeros([32]), name='conv2_biases')

        conv3_weights = tf.Variable(
            tf.truncated_normal([3, 3, 32, 32],
                                stddev=0.1,
                                seed=SEED), name='conv3_weights')
        conv3_biases = tf.Variable(tf.constant(0.1, shape=[32]), name='conv3_biases')

        conv4_weights = tf.Variable(
            tf.truncated_normal([3,3,32,64],
                                stddev=0.1,
                                seed=SEED), name='conv4_weights')
        conv4_biases = tf.Variable(tf.constant(0.1,shape=[64]), name='conv4_biases')
       


        fc1_weights = tf.Variable(  # fully connected, depth 512.
            #originally: int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 80) , now 320
            tf.truncated_normal([FC1_WIDTH, 64],
                                stddev=0.1,
                                seed=SEED), name='fc1_weights')
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='fc1_biases')

        fc2_weights = tf.Variable(  # fully connected, depth 64.
            tf.truncated_normal([64, NUM_LABELS],
                                stddev=0.1,
                                seed=SEED), name='fc2_weights')
        fc2_biases  = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]), name='fc2_biases')



    elif init_type == 'xavier':
        conv1_weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        conv1_weights = tf.Variable( conv1_weights_init(shape=[5, 5, NUM_CHANNELS, 16]), name='conv1_weights')
        conv1_biases = tf.Variable(tf.constant(0.01, shape=[16]), name='conv1_biases')

        conv2_weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        conv2_weights = tf.Variable( conv2_weights_init(shape=[3, 3, 16, 32]), name='conv2_weights')
        conv2_biases = tf.Variable(tf.constant(0.01, shape=[32]), name='conv2_biases')

        conv3_weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        conv3_weights = tf.Variable( conv3_weights_init(shape=[3, 3, 32, 32]), name='conv3_weights')
        conv3_biases = tf.Variable(tf.constant(0.1, shape=[32]), name='conv3_biases')

        conv4_weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        conv4_weights = tf.Variable( conv4_weights_init(shape=[3,3,32,64]), name='conv4_weights')
        conv4_biases = tf.Variable(tf.constant(0.1,shape=[64]), name='conv4_biases')


        fc1_weights_init = tf.contrib.layers.xavier_initializer()
        fc1_weights = tf.Variable(  fc1_weights_init(shape=[FC1_WIDTH, 64]), name='fc1_weights')
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='fc1_biases')

        fc2_weights_init = tf.contrib.layers.xavier_initializer()
        fc2_weights = tf.Variable( fc2_weights_init(shape=[64, NUM_LABELS]), name='fc2_weights')
        fc2_biases  = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]), name='fc2_biases')
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
        cropped = img_crop_context(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE,CONTEXT_ADDITIVE_FACTOR)
        data = numpy.asarray(cropped)

        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

        return img_prediction

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)
        img_prediction = get_prediction(img)
        cimg = concatenate_images(img, img_prediction)

        return cimg

    def gtruth_pred(filename,image_idx):
        imageid = image_idx
        fname = "test_"+str(image_idx)

        image_filename = filename +fname+"/"+fname+".png"
        img = mpimg.imread(image_filename)
        img_prediction = get_prediction(img)
        cimg = concatenate_images(img,img_prediction)
        return cimg

    def generate_real_out(filename,image_idx):
        imageid = image_idx
        fname = "test_"+str(image_idx)

        image_filename = filename +fname+"/"+fname+".png"
        img = mpimg.imread(image_filename)
        #img = Image.open(image_filename)
        #downscaled = img.resize((200,200)) #HARDCODED
        #img = numpy.asarray(downscaled)
        img_prediction = get_prediction(img)
        return img_prediction


    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg

    def otruth_pred(filename,image_idx):
         #imageid = "satImage_%.3d" % image_idx
        fname = "test_"+str(image_idx)
        image_filename = filename + fname+"/"+fname + ".png"
        img = mpimg.imread(image_filename)
        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)
        return oimg


    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        data = tf.cast(data, dtype=tf.float32)
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1], #changed to stride=4
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
      
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool1 = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

       

        conv2 = tf.nn.conv2d(pool1,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
       
        pool2 = tf.nn.max_pool(relu2,
                                ksize = [1,2,2,1],
                                strides=[1,2,2,1],
                                padding='SAME')

        
        conv3 = tf.nn.conv2d(pool2,
                            conv3_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))
        
        pool3 = tf.nn.max_pool(relu3,
                                ksize=[1,2,2,1],
                                strides = [1,2,2,1],
                                padding='SAME')
        
        conv4 = tf.nn.conv2d(pool3,
                            conv4_weights,
                            strides=[1,1,1,1],
                            padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,conv4_biases))
       
        pool4 = tf.nn.max_pool(relu4,
                                ksize=[1,2,2,1],
                                strides=[1,2,2,1],
                                padding='SAME')
        

       
        if train:
            print("relu1: "+str(relu.get_shape()))
            print("Pool1: "+str(pool1.get_shape()))
            print("relu2: "+str(relu2.get_shape()))
            print("pool2: "+str(pool2.get_shape()))
            print("relu3: "+str(relu3.get_shape()))
            print("pool3: "+str(pool3.get_shape()))
            print("relu4: "+str(relu4.get_shape()))
            print("pool4: "+str(pool4.get_shape()))

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        relu_shape = pool4.get_shape().as_list()
        reshape = tf.reshape(
            pool4,
            [relu_shape[0], relu_shape[1] * relu_shape[2] * relu_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        
        #hidden1 = tf.layers.dense(inputs = reshape,units=1024,activation = tf.nn.relu)
        hidden1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        
        #hidden2 = tf.nn.relu(tf.matmul(hidden1,fc2_weights) + fc2_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden1, 0.9, seed=SEED)

        out = tf.matmul(hidden1, fc2_weights) + fc2_biases
        #out = tf.layers.dense(inputs = hidden1, units=2)
        if train == True:
            summary_id = '_0'
            s_data = get_image_summary(data)
            filter_summary0 = tf.summary.image('summary_data' + summary_id, s_data)
            s_conv = get_image_summary(conv)
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
    logits = model(train_data_node, MODE=='train') # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=train_labels_node))
    tf.summary.scalar('loss', loss)

    all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, 
                    conv3_weights, conv3_biases, conv4_weights, conv4_biases, 
                    fc1_weights, fc1_biases, fc2_weights, fc2_biases]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 
                    'conv3_weights','conv3_biases', 'conv4_weights','conv4_biases',
                    'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases']
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)
    
    # L2 regularization for the fully connected parameters.
    regula