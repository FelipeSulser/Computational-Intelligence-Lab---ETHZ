import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

NUM_CHANNELS = 3
PIXEL_DEPTH = 255

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

    return np.asarray(imgs)


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


def img_float_rescale(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) )
    return rimg    


def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    test_dir = 'test_set_images/'

    # Extract it into np arrays.
    TRAINING_SIZE = 100
    train_data = extract_data(train_data_filename, TRAINING_SIZE)

    N_TEST_FILES = len([name for name in os.listdir(test_dir) if name.startswith('test_')])
    TEST_FILES = [name for name in os.listdir(test_dir) if name.startswith('test_')]

    print('train_data.shape: ', train_data.shape)
    train_size = train_data.shape[0]
    IMG_SIZE = train_data.shape[1]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed all of training data using 
    # the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(3, IMG_SIZE, IMG_SIZE, 1))

    # Here define the constant Convolution filter
    # it must have shape (x, x, NUM_CHANNELS, 1)
    corner = -1
    edge = -1
    center = 1
    
    conv_filter1 = tf.constant([ 
    	[corner,edge,corner],
    	[edge,center,edge],
    	[corner,edge,corner],
    ], tf.float32)
    print('conv_filter1 shape: ', conv_filter1.get_shape())

    conv_filter2 = tf.reshape(conv_filter1, [3, 3, 1, 1])
    print('conv_filter2 shape: ', conv_filter2.get_shape())

    # conv_filter3 = tf.concat([conv_filter2,conv_filter2,conv_filter2], 0)
    # print('conv_filter3 shape: ', conv_filter3.get_shape())

    # conv_filter4 = tf.concat([conv_filter3,conv_filter3,conv_filter3], 2)
    # print('conv_filter4 shape: ', conv_filter4.get_shape())


    def model(data):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv_filter2,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
  
        return conv

    #convoluted = model(train_data_node, True)

    save_dir = 'training/sharp_training/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir) 

    with tf.Session() as s:
    	tf.initialize_all_variables().run()
    	#feed_dict = {train_data_node: train_data}

    	#result = s.run([convoluted], feed_dict=feed_dict)
    	#result = np.asarray(result)[0]
    	#print('result.shape: ', result.shape)

    	for i in range(0,TRAINING_SIZE):
    		data_node_r = np.reshape(train_data[i,:,:,0], (1,400,400,1))
    		data_node_g = np.reshape(train_data[i,:,:,1], (1,400,400,1))
    		data_node_b = np.reshape(train_data[i,:,:,2], (1,400,400,1))
    		data_node = np.concatenate((data_node_r,data_node_g,data_node_b), 0)
    		#print('data_node: ', data_node.shape)


    		feed_dict = {train_data_node: data_node}
    		res = s.run(model(train_data_node), feed_dict=feed_dict)
    		res = np.asarray(res)
    		#print(res.shape)
    		res_r = res[0,:,:,:]
    		res_g = res[1,:,:,:]
    		res_b = res[2,:,:,:]
    		res2 = np.concatenate((res_r,res_g,res_b), axis=2)
    		#print(res2.shape)
    		res2 = img_float_to_uint8(res2)


    		Image.fromarray(res2).convert('RGB').save(save_dir+'satImage_' + '%.3d' %(i+1) + '.png')



if __name__ == '__main__':
	tf.app.run()   	
