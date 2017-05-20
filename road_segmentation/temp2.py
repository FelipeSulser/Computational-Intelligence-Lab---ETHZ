import os 
import numpy as np
import scipy as scp
import pylab as pyl
from scipy.misc import imsave as  im_save
import matplotlib.pyplot as plt
import pywt
import matplotlib.image as mpimg
from numpy import linalg
import warnings

PIXEL_DEPTH = 255

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    print(np.min(rimg))
    print(np.max(rimg))
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH)
    print(np.min(rimg))
    print(np.max(rimg))
    rimg = rimg.round().astype(np.uint8)
    return rimg


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
            #downscaled = np.asarray(downscaled)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')



    return np.asarray(imgs)

STARTING_ID = 1
TRAINING_SIZE = 200
data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
train_data_filename = data_dir + 'images/'		

train_data = extract_data(train_data_filename, TRAINING_SIZE, STARTING_ID, 0)




mean_img = np.mean(train_data, axis=0)

print('MEAN: ',mean_img.shape)
save_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
im_save(save_dir+'mean_img.png', mean_img)


# for i in range(train_data.shape[0]):
# 	example_img = train_data[i,:,:,:] - mean_img
# 	example_img = img_float_to_uint8(example_img)
# 	plt.imshow(example_img)
# 	plt.show()

