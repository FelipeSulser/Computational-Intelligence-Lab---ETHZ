from scipy import ndimage
import matplotlib.image as mpimg
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy
from skimage import morphology
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import feature

from skimage import data, img_as_float
from skimage import exposure
from skimage import filters

MODE = 'test'



@adapt_rgb(hsv_value)
def scharr_each(image):
    return filters.scharr(image)

if MODE == 'train':
    save_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/sharp_training/'
elif MODE == 'test':
    save_dir = (os.path.dirname(os.path.realpath(__file__)))+'/test_set_images_filtered/'

if not os.path.isdir(save_dir):
	os.mkdir(save_dir) 



imgs = []
if MODE == 'train':
    data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
    train_data_filename = data_dir + 'images/'
    range_max = 201
elif MODE == 'test':
    train_data_filename = (os.path.dirname(os.path.realpath(__file__)))+'/test_set_images/'    
    range_max = 51

for i in range(1, range_max):
    imageid = "satImage_%.3d" % i
    if MODE == 'train':
        image_filename = train_data_filename + imageid + ".png"
    elif MODE == 'test':  
        fname = "test_"+str(i)  
        image_filename = train_data_filename + fname + '/' + fname + ".png"

    if os.path.isfile(image_filename):
        img = mpimg.imread(image_filename)
        p2, p98 = np.percentile(img, (2, 98))
        img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
        img_eq = exposure.equalize_hist(img)
        img_eq = img_as_float(img_eq)
        if MODE == 'train':
            scipy.misc.imsave(save_dir+imageid+".png",img_eq)
        elif MODE == 'test':
            dist_fir = save_dir+fname+'/'
            if not os.path.isdir(dist_fir):
                os.mkdir(dist_fir) 
            scipy.misc.imsave(dist_fir+fname+".png",img_eq) 






#edges = scharr_each(img)
#plt.imshow(edges)
#plt.show()





       



