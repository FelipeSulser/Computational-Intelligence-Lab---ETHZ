import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float, color
from skimage.util import random_noise

img_dir = 'training/images/'
i = 1
img_filename = img_dir + ('satImage_%.3d' % i)+'.png'

original = mpimg.imread(img_filename)

sigma = 0.155
noisy = random_noise(original, var=sigma**2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharex=True,
                       sharey=True, subplot_kw={'adjustable': 'box-forced'})

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print("Estimated Gaussian noise standard deviation = {}".format(sigma_est))


ax[1].imshow(denoise_wavelet(original, multichannel=True, convert2ycbcr=True))
ax[1].axis('off')
ax[1].set_title('Wavelet denoising\nin YCbCr colorspace')
ax[0].imshow(original)
ax[0].axis('off')
ax[0].set_title('Original')

fig.tight_layout()

plt.show()