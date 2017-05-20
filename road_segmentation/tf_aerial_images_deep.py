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


VALIDATION_SIZE = 1  # Size of the validation set.

SEED = None  # Set to None for random seed.
#TODO change batch size
BATCH_SIZE = 32 # 64
#TODO change epoch number
NUM_EPOCHS = 9
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
DOWNSCALE = 1


MODE = 'train' # 'train' or 'predict'
STARTING_ID = 1 # 21, 41...
TRAINING_SIZE = 20


init_type = 'xavier'

TEST_START_ID = 1 
TEST_SIZE = 50

LOGGING = False
 



# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!

CONTEXT_ADDITIVE_FACTOR = 19 #patch context increased by 2x2, so a 8x8 patch becomes a 16x15
IMG_PATCH_SIZE = 12 #should be at least dividor of 608
CONTEXT_PATCH = IMG_PATCH_SIZE+2*CONTEXT_ADDITIVE_FACTOR #in this case window is 16x16

 
if CONTEXT_PATCH == 40:
    FC1_WIDTH = 576
elif CONTEXT_PATCH == 64:
    FC1_WIDTH = 1024
elif CONTEXT_PATCH == 32:
    FC1_WIDTH = 2048
elif CONTEXT_PATCH == 50:
    FC1_WIDTH = 3136    
else:
    FC1_WIDTH = 3136 # TODO 
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


    validation_data = extract_data(train_data_filename,VALIDATION_SIZE,200-VALIDATION_SIZE,CONTEXT_ADDITIVE_FACTOR)
    validation_labels = extract_labels(train_labels_filename,VALIDATION_SIZE,200-VALIDATION_SIZE,CONTEXT_ADDITIVE_FACTOR)
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
            tf.truncated_normal([3, 3, 32, 48],
                                stddev=0.1,
                                seed=SEED), name='conv3_weights')
        conv3_biases = tf.Variable(tf.constant(0.01, shape=[48]), name='conv3_biases')

        conv4_weights = tf.Variable(
            tf.truncated_normal([3,3,48,64],
                                stddev=0.1,
                                seed=SEED), name='conv4_weights')
        conv4_biases = tf.Variable(tf.constant(0.01,shape=[64]), name='conv4_biases')
        


        fc1_weights = tf.Variable(  # fully connected, depth 512.
            #originally: int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 80) , now 320
            tf.truncated_normal([FC1_WIDTH, 64],
                                stddev=0.1,
                                seed=SEED), name='fc1_weights')
        fc1_biases = tf.Variable(tf.constant(0.01, shape=[64]), name='fc1_biases')

        fc2_weights = tf.Variable(  # fully connected, depth 64.
            tf.truncated_normal([64, 32],
                                stddev=0.1,
                                seed=SEED), name='fc2_weights')
        fc2_biases  = tf.Variable(tf.constant(0.01, shape=[32]), name='fc2_biases')

        fc3_weights = tf.Variable(  # fully connected, depth 64.
            tf.truncated_normal([32, NUM_LABELS],
                                stddev=0.1,
                                seed=SEED), name='fc3_weights')
        fc3_biases  = tf.Variable(tf.constant(0.01, shape=[NUM_LABELS]), name='fc3_biases')


 
    elif init_type == 'xavier': 
        conv1_weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        conv1_weights = tf.Variable( conv1_weights_init(shape=[10, 10, NUM_CHANNELS, 16]), name='conv1_weights')
        conv1_biases = tf.Variable(tf.constant(0.001, shape=[16]), name='conv1_biases')

        conv2_weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        conv2_weights = tf.Variable( conv2_weights_init(shape=[5, 5, 16, 32]), name='conv2_weights')
        conv2_biases = tf.Variable(tf.constant(0.001, shape=[32]), name='conv2_biases')

        conv3_weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        conv3_weights = tf.Variable( conv3_weights_init(shape=[5, 5, 32, 48]), name='conv3_weights')
        conv3_biases = tf.Variable(tf.constant(0.01, shape=[48]), name='conv3_biases')

        conv4_weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        conv4_weights = tf.Variable( conv4_weights_init(shape=[3,3,48,64]), name='conv4_weights')
        conv4_biases = tf.Variable(tf.constant(0.01,shape=[64]), name='conv4_biases')


        fc1_weights_init = tf.contrib.layers.xavier_initializer()
        fc1_weights = tf.Variable(  fc1_weights_init(shape=[FC1_WIDTH, 64]), name='fc1_weights')
        fc1_biases = tf.Variable(tf.constant(0.01, shape=[64]), name='fc1_biases')

        fc2_weights_init = tf.contrib.layers.xavier_initializer()
        fc2_weights = tf.Variable( fc2_weights_init(shape=[64, 32]), name='fc2_weights')
        fc2_biases  = tf.Variable(tf.constant(0.01, shape=[32]), name='fc_biases')
 
        fc3_weights_init = tf.contrib.layers.xavier_initializer()
        fc3_weights = tf.Variable( fc2_weights_init(shape=[32, NUM_LABELS]), name='fc3_weights')
        fc3_biases  = tf.Variable(tf.constant(0.01, shape=[NUM_LABELS]), name='fc3_biases')
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
        
        
        conv4 = tf.nn.conv2d(relu3,
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

        hidden2 = tf.nn.relu(tf.matmul(hidden1,fc2_weights)+fc2_biases)
        
        #hidden2 = tf.nn.relu(tf.matmul(hidden1,fc2_weights) + fc2_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden1, 0.5, seed=SEED)
            hidden2 = tf.nn.dropout(hidden2,0.5,seed=SEED)

        out = tf.matmul(hidden2, fc3_weights) + fc3_biases
        #out = tf.layers.dense(inputs = hidden1, units=2)
        if LOGGING == True:
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
                    fc1_weights, fc1_biases, fc2_weights, fc2_biases,fc3_weights,fc3_biases]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 
                    'conv3_weights','conv3_biases', 'conv4_weights','conv4_biases',
                    'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases','fc3_weights','fc3_biases']
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)
    
    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) + 
                    tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases))
    # Add the regularization term to the loss.
    loss += 5e-5 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0, name='batch')
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    #learning_rate = 0.01
    tf.summary.scalar('learning_rate', learning_rate)
    
    
    #AdamOptimizer - adaptative momentum
    # learning_rate: 1e-4 is very often used. ADAM chooses itsself a learning rate, 
    #               so tf.train.exponential_decay might not be a good idea
    # epsilon: 0.1 as recommended for imagenet
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, 
            beta1= 0.9, beta2 = 0.999,epsilon=0.1).minimize(loss,global_step=batch)


    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    #train_all_prediction = tf.nn.softmax(model(train_all_data_node))

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


    # Create a local session to run this computation.
    #tf.get_default_graph().finalize()
    with tf.Session() as s:

        if MODE == 'predict':
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model_deep.ckpt")
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
            for i in range(TEST_START_ID, TEST_START_ID+TEST_SIZE):
                print("Prediction for img: "+str(i))

                # read image
                fname = "test_"+str(i)
                image_filename = test_set_dir + fname+"/"+fname + ".png"
                img = mpimg.imread(image_filename)
                # predict label
                img_prediction = get_prediction(img)
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
                saver.restore(s, FLAGS.train_dir + "/model_deep.ckpt")



            # Build the summary operation based on the TF collection of Summaries.
            if LOGGING:
                summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                                    graph=s.graph)
            
            # Loop through training steps.
            print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))
            training_indices = range(train_size)

            for iepoch in range(num_epochs):

                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)
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

                            _, l, lr, predictions = s.run(
                                [ optimizer, loss, learning_rate, train_prediction],
                                feed_dict=feed_dict)

                        print ('Epoch %.2f' % (iepoch))
                        print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                # Save the variables to disk.
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)


            data_node_validation = tf.constant(validation_data)
            output_validation = tf.nn.softmax(model(data_node_validation))
            print("VALIDATION:")            
            output_prediction_validation = s.run(output_validation)

            



         


if __name__ == '__main__':

    tf.app.run()
