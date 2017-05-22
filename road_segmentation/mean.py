import os 
import numpy as np
import scipy as scp
import pylab as pyl
from scipy.misc import imsave as  im_save
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import linalg
import warnings

PIXEL_DEPTH = 255
CONTEXT_ADDITIVE_FACTOR = 24 
IMG_PATCH_SIZE = 16 
CONTEXT_PATCH = IMG_PATCH_SIZE+2*CONTEXT_ADDITIVE_FACTOR

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    print(np.min(rimg))
    print(np.max(rimg))
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH)
    print(np.min(rimg))
    print(np.max(rimg))
    rimg = rimg.round().astype(np.uint8)
    return rimg


# Extract patches from a given image
# the patch itself should be (w+2*context_factor,h+2*context_factor)
# but the base used for labeling should be (w,h)
def img_crop_context(im, w, h,context_factor):

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

    img_patches = [img_crop_context(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE,context_factor) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    data = np.asarray(data)
    return data



STARTING_ID = 1
TRAINING_SIZE = 214
data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/images_shuffled/'
train_data_filename = data_dir #+ 'images/'		

train_data = extract_data(train_data_filename, TRAINING_SIZE, STARTING_ID, CONTEXT_ADDITIVE_FACTOR)




mean_patch = np.mean(train_data, axis=0)

print('MEAN: ',mean_patch.shape)
save_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
im_save(save_dir+'mean_patch_64.png', mean_patch)

'''
for i in range(train_data.shape[0]):
    f, axarr = plt.subplots(1,2)
    orig = img_float_to_uint8(train_data[i,:,:,:])
    axarr[0].imshow(orig)

    example_img = train_data[i,:,:,:] - mean_patch
    #example_img = img_float_to_uint8(example_img)

    
    axarr[1].imshow(example_img)


    plt.show()

'''