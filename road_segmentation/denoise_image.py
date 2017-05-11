#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
from scipy import ndimage
from PIL import Image
import numpy
from skimage import restoration

IMG_PATCH_SIZE = 8 #8x8

def denoise(img, weight=0.1, eps=1e-3, num_iter_max=100):
    """Perform total-variation denoising on a grayscale image.
     
    Parameters
    ----------
    img : array
        2-D input data to be de-noised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more
        de-noising (at the expense of fidelity to `img`).
    eps : float, optional
        Relative difference of the value of the cost
        function that determines the stop criterion.
        The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    num_iter_max : int, optional
        Maximal number of iterations used for the
        optimization.
 
    Returns
    -------
    out : array
        De-noised array of floats.
     
    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """
    u = np.zeros_like(img)
    px = np.zeros_like(img)
    py = np.zeros_like(img)
     
    nm = np.prod(img.shape[:2])
    tau = 0.125
     
    i = 0
    while i < num_iter_max:
        u_old = u
        # x and y components of u's gradient
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u
        # update the dual variable
        px_new = px + (tau / weight) * ux
        py_new = py + (tau / weight) * uy
        norm_new = np.maximum(1, np.sqrt(px_new **2 + py_new ** 2))
        px = px_new / norm_new
        py = py_new / norm_new
        # calculate divergence
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)
        # update image
        u = img + weight * div_p
        # calculate error
        error = np.linalg.norm(u - u_old) / np.sqrt(nm)
        if i == 0:
            err_init = error
            err_prev = error
        else:
            # break if error small enough
            if np.abs(err_prev - error) < eps * err_init:
                break
            else:
                e_prev = error
             
        # don't forget to update iterator
        i += 1
 
    return u


def linearsmoothen(img,num_neigh):
    resimg = ndimage.median_filter(img, num_neigh)
    #resimg = ndimage.filters.gaussian_filter(img, num_neigh, mode='nearest')
    resimg[resimg > 0.5] = 1.0
    resimg[resimg <= 0.5] = 0.0
    return resimg

def morphologysmoothen(img,num_neigh):
    (width, height) = img.size
    greyscale_map = list(img.getdata())
    greyscale_map = numpy.array(greyscale_map)

    img = greyscale_map.reshape((height, width))
    intermed = ndimage.binary_erosion(img,structure=(np.zeros((10,10))))
    resimg = ndimage.binary_propagation(intermed,mask=img)
    return resimg

def tvsmoothen(img,num_neigh):
    (width, height) = img.size
    greyscale_map = list(img.getdata())
    greyscale_map = numpy.array(greyscale_map)

    img = greyscale_map.reshape((height, width))
    U = denoise(img,weight=num_neigh)
    #print(U)
    U[U<100] = 0
    U[U != 0] = 255
    return U

if __name__ == '__main__':
    output_dir = 'myprediction/denoised/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    image_filenames = []
    for i in range(1, 51):
        image_filename = 'myprediction/result/prediction_' +  str(i) + '.png'
        image_file = Image.open(image_filename) # open colour image
        image_file = image_file.convert('L') # convert image to black and white
        img = tvsmoothen(image_file,300)
        img = img.astype(float)
        Image.fromarray(img).convert('RGB').save(output_dir+"prediction_"+str(i)+".png")
