
import numpy as np
import scipy as scp
import pylab as pyl

import matplotlib.pyplot as plt
import pywt
import matplotlib.image as mpimg
from nt_toolbox.general import *
from nt_toolbox.signal import *
from numpy import linalg
from nt_toolbox.compute_wavelet_filter import *
import warnings





img_dir = 'training/images/'
i = 1
img_filename = img_dir + ('satImage_%.3d' % i)+'.png'

img = mpimg.imread(img_filename)



warnings.filterwarnings('ignore')


Jmin = 1
img = img[:,:,0]
n = 512
img = rescale(img, n)
h = compute_wavelet_filter("Daubechies",8)
print(h)
fW = perform_wavortho_transf(img, Jmin, + 1, h)

plt.figure(figsize = (8,8))

plot_wavelet(fW, Jmin)
plt.title('Wavelet coefficients')

plt.show()