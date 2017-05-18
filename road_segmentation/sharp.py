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

def extract_data(directory, filenames):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for ff in filenames:
        full_path = directory + ff
        if os.path.isfile(full_path):
            print ('Loading ' + full_path)
            img = mpimg.imread(full_path)
            print(img.shape)
            imgs.append(img)
        else:
            print ('File ' + full_path + ' does not exist')

    return np.asarray(imgs)


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


def img_float_rescale(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) )
    return rimg    



def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'predictions_test/result/'
    out_dir = 'predictions_test/result_smooth/'

    # Extract it into np arrays.
    

    
    IN_FILES = [name for name in os.listdir(data_dir) if name.endswith('.png')]

    #train_data = extract_data(data_dir, IN_FILES)

    #print('train_data.shape: ', train_data.shape)
    IMG_SIZE = 608

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
    center = 2
    filter_size = 7
    filter_mat = np.ones((filter_size,filter_size))

    conv_filter1 = tf.constant( filter_mat, tf.float32)

    print('conv_filter1 shape: ', conv_filter1.get_shape())

    conv_filter2 = tf.reshape(conv_filter1, [filter_mat.shape[0], filter_mat.shape[1], 1, 1])
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

    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir) 

    with tf.Session() as s:
        tf.initialize_all_variables().run()
        #feed_dict = {train_data_node: train_data}

        #result = s.run([convoluted], feed_dict=feed_dict)
        #result = np.asarray(result)[0]
        #print('result.shape: ', result.shape)

        for ff in IN_FILES:
            full_path = data_dir + ff
            if os.path.isfile(full_path):
                print ('Loading ' + full_path)
                img = mpimg.imread(full_path)
                print('img: ',img.shape)
                data_node_r = np.reshape(img[:,:,0], (1,IMG_SIZE,IMG_SIZE,1))
                data_node_g = np.reshape(img[:,:,1], (1,IMG_SIZE,IMG_SIZE,1))
                data_node_b = np.reshape(img[:,:,2], (1,IMG_SIZE,IMG_SIZE,1))
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


                Image.fromarray(res2).convert('RGB').save(out_dir+ff)
                print('File %s saved!' %(out_dir+ff))

            else:
                print ('File ' + full_path + ' does not exist')


if __name__ == '__main__':
	tf.app.run()   	