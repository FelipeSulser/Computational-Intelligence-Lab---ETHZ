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
import pywt
import math


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
        img -= np.mean(img,axis=0)
        X = img.reshape(-1, 3)
        cov = np.dot(X.T,X) / X.shape[0]


        U,S,V = np.linalg.svd(cov)

        Xrot = np.dot(X,U)
        Xwhite = Xrot/np.sqrt(S+1e-5)
        Xwhite = Xwhite.reshape((400,400,3))
        print(Xwhite.shape)
        #X = img
        #cov = np.dot(X.T,X)/X.shape[0]
        #U,S,V = np.linalg.svd(cov)
        #Xrot = np.dot(X,U)
        #Xwhite = Xrot/np.sqrt(S+1e-5)
        #f, axarr = plt.subplots(2,2)
        #axarr[0,0].imshow(img[:,:,0])
        #axarr[0,1].imshow(img[:,:,1])
        #axarr[1,0].imshow(img[:,:,2])
        #plt.show()

        wavelet = pywt.Wavelet('haar')
        levels  = int( math.floor( np.log2(img[:,:,0].shape[0]) ) )
        noiseSigma = 16.0
        
        WaveletCoeffs = pywt.wavedec2( img[:,:,0], wavelet, level=levels)
        threshold = noiseSigma*math.sqrt(2*np.log2(img[:,:,0].size))
        NewWaveletCoeffs = map (lambda x: pywt.thresholding.soft(x,threshold),
        WaveletCoeffs)
        NewImage = pywt.waverec2( NewWaveletCoeffs, wavelet)
        plt.imshow(NewImage)
        plt.show()
        scipy.misc.imsave(save_dir+imageid+".png",img)






#edges = scharr_each(img)
#plt.imshow(edges)
#plt.show()





       



