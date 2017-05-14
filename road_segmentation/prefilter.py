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


@adapt_rgb(hsv_value)
def scharr_each(image):
    return filters.scharr(image)

save_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/sharp_training/'
if not os.path.isdir(save_dir):
	os.mkdir(save_dir) 



imgs = []
data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/' 

for i in range(1, 201):
    imageid = "satImage_%.3d" % i
    image_filename = train_data_filename + imageid + ".png"

    if os.path.isfile(image_filename):
    	img = mpimg.imread(image_filename)
    	p2, p98 = np.percentile(img, (2, 98))
    	img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    	img_eq = exposure.equalize_hist(img)
    	img_eq = img_as_float(img_eq)
    	scipy.misc.imsave(save_dir+imageid+".png",img_eq)






#edges = scharr_each(img)
#plt.imshow(edges)
#plt.show()





       



