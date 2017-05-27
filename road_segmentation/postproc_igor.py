import math
import os

import numpy as np
import tensorflow as tf

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from scipy.misc import imsave as  im_save










NUM_CHANNELS = 3
PIXEL_DEPTH = 255

NEIGHBOOR_TO_CONSIDER = 8
BALCK_TH = int(0.75 * NEIGHBOOR_TO_CONSIDER)
WHITE_TH = int(0.25 * NEIGHBOOR_TO_CONSIDER)
 
    
IMG_SIZE = 608


def remove_filtering_neighbors(img,black_threshold, block_size = 16):
    #img is b&w array with 0 or 1
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    
    numblockwidth = int(imgwidth/block_size)
    numblockheight = int(imgheight/block_size)

    for i in range(1,numblockwidth-2):
        for j in range(1, numblockheight-2):
            pixel_i = i*block_size
            pixel_j = j*block_size


            if img[pixel_i,pixel_j] == 0: #if patch is black
                #if not surrounded by 3 cut it
                neighbors = np.zeros(8)

                #black is 0

                neighbors[0] = img[pixel_i-block_size,pixel_j]
                neighbors[1] = img[pixel_i+block_size,pixel_j]
                neighbors[2] = img[pixel_i,pixel_j-block_size]
                neighbors[3] = img[pixel_i,pixel_j+block_size]
                neighbors[4] = img[pixel_i-block_size,pixel_j-block_size]
                neighbors[5] = img[pixel_i-block_size,pixel_j+block_size]
                neighbors[6] = img[pixel_i+block_size,pixel_j-block_size]
                neighbors[7] = img[pixel_i+block_size,pixel_j+block_size]

                sum_val = np.sum(neighbors)
                if(sum_val > black_threshold):
                    #print('  Block repainted to BLACK!')
                    for xx in range(0,block_size):
                        for yy in range(0,block_size):
                            img[pixel_i+xx,pixel_j+yy] = 1.0
            else: #white patch 1
                #if not surrounded by 3 cut it
                neighbors = np.zeros(8)

                #black is 0

                neighbors[0] = img[pixel_i-block_size,pixel_j]
                neighbors[1] = img[pixel_i+block_size,pixel_j]
                neighbors[2] = img[pixel_i,pixel_j-block_size]
                neighbors[3] = img[pixel_i,pixel_j+block_size]
                neighbors[4] = img[pixel_i-block_size,pixel_j-block_size]
                neighbors[5] = img[pixel_i-block_size,pixel_j+block_size]
                neighbors[6] = img[pixel_i+block_size,pixel_j-block_size]
                neighbors[7] = img[pixel_i+block_size,pixel_j+block_size]


                sum_val = np.sum(neighbors)
                wh_threshold = NEIGHBOOR_TO_CONSIDER-black_threshold+1
                if(sum_val < wh_threshold):
                    #print('  Block repainted to WHITE!')
                    for xx in range(0,block_size):
                        for yy in range(0,block_size):
                            img[pixel_i+xx,pixel_j+yy] = 0.0


    return img


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def binarize(img,block_size,threshold):
    
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    
    numblockwidth = int(imgwidth/block_size)

    numblockheight = int(imgheight/block_size)

    for i in range(0,numblockwidth):
        for j in range(0, numblockheight):
            pixel_i = i*block_size
            pixel_j = j*block_size
            avg = np.mean(img[pixel_i:pixel_i+block_size,pixel_j:pixel_j+block_size])
            if(avg > threshold):
                img[pixel_i:pixel_i+block_size,pixel_j:pixel_j+block_size] = 1
            else:
                img[pixel_i:pixel_i+block_size,pixel_j:pixel_j+block_size] = 0

    return img


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


def fill_rows_and_cols(img, block_size=16, missing_blocks=3):
    row_sums = np.sum(img, axis=1)
    col_sums = np.sum(img, axis=0)

    n_blocks_per_rc = math.ceil(IMG_SIZE / block_size)
    rc_threshold = IMG_SIZE - (missing_blocks+1) * block_size


    for row, row_sum in enumerate(row_sums):
        if row_sum > rc_threshold:
            # number of road patches is bigger than the minimum
            img[row,:] = 1

    for col, col_sum in enumerate(col_sums):
        if col_sum > rc_threshold:
            # number of road patches is bigger than the minimum
            img[:,col] = 1
    
    return img


def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'predictions_test/result_azure_deep/result/'
    out_dir = 'predictions_test/result_smooth_bin/'

    # Extract it into np arrays.
    
    
    IN_FILES = [name for name in os.listdir(data_dir) if name.endswith('.png')]

    #train_data = extract_data(data_dir, IN_FILES)

    #print('train_data.shape: ', train_data.shape)
    

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed all of training data using 
    # the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(1, IMG_SIZE, IMG_SIZE, 1))

    # Here define the constant Convolution filter
    # it must have shape (x, x, NUM_CHANNELS, 1)
    corner = -1
    edge = -1
    center = 2
    filter_size = 9
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
                img = rgb2gray(img)

                print('img: ',img.shape)
                data_node = np.reshape(img[:,:], (1,IMG_SIZE,IMG_SIZE,1))
                #print('data_node: ', data_node.shape)


                feed_dict = {train_data_node: data_node}
                res = s.run(model(train_data_node), feed_dict=feed_dict)
                res = np.asarray(res)

                res = np.reshape(res, (res.shape[1], res.shape[2]))
                print('res: ', res.shape)
                res = img_float_to_uint8(res)
                #print(res)
                res = res/255
                res_igor = binarize(res, 16, 0.4)
                res_igor = remove_filtering_neighbors(res_igor,7,block_size=16)

                res_igor = fill_rows_and_cols(res_igor, missing_blocks=3)
                # img_tv = 1-img
                # tv_denoise = denoise_tv_chambolle(img_tv, weight=10)
                # tv_denoise_bw = binarize(tv_denoise,16,0.7)
                # tv_denoise_bw = remove_filtering_neighbors(tv_denoise_bw,7,block_size=16)

                # tv_denoise_bw = 1-tv_denoise_bw
                # f, axarr = plt.subplots(2,2)
                # axarr[0,0].imshow(img)
                # axarr[0,1].imshow(tv_denoise_bw)
                # axarr[1,0].imshow(res_igor)
                
                # plt.show()

                im_save(out_dir+ff, res_igor)
                print('File %s saved!' %(out_dir+ff))
                print()

            else:
                print ('File ' + full_path + ' does not exist')


if __name__ == '__main__':
	tf.app.run()   	