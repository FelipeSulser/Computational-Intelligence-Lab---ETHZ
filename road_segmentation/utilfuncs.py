'''

Authors:

- Igor Pesic
- Felipe Sulser
- Minesh Patel

Common functions used all over the code that are needed

'''
import numpy as np
import matplotlib.image as mpimg

NEIGHBOOR_TO_CONSIDER = 8

#converts a 3D image to a 2D image that is in grayscale
def rgb2gray(rgb):
    """
    Function that creates a 2D image in grayscale given 3D RGB image
    
    Input:
    rgb: 3D image in RGB
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# binarizes the image given a threshold.
# This converts the image from grayscale (0-255) to black&white (1 bit only)
def binarize(img,block_size,threshold):
    """
    Function that converts grayscale image to black & white (1 bit) image

    Input:
    img: 2D image in grayscale
    block_size: Size of the block used in the road segmentation (usually 16)
    threshold: avg value that marks the difference between 0 or 1 (black or white)

    """
    
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

# Function that changes color of pixel i,j given its 8 neighbors
def remove_filtering_neighbors(img,black_threshold, block_size = 16):
    """
    Function that changes color of block i,j given its 8 neighbors.
    If the number of different colored neighbors is greater than the threshold,
    color is changed

    Input:
    img: a b&w image (not grayscale)
    black_threshold: number of white neighbors to change a black patch to white
    block_size: Block_size used in the segmentation (usually 16)

    """
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    
    numblockwidth = int(imgwidth/block_size)
    numblockheight = int(imgheight/block_size)

    for i in range(1,numblockwidth-1):
        for j in range(1, numblockheight-1):
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
                    #repaint block
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
                wh_threshold = NEIGHBOOR_TO_CONSIDER-black_threshold
                if(sum_val < wh_threshold):
                    for xx in range(0,block_size):
                        for yy in range(0,block_size):
                            img[pixel_i+xx,pixel_j+yy] = 0.0
            


    return img

def fill_rows_and_cols(img, block_size=16, missing_blocks=3):
    """
    Function that changes the color of patches 
    if the number of distinct patches per column
    or per row is less than "missing_blocks"

    Input:
    img: a b&w image (not grayscale)
    block_size: block_size used in the segmentation (usually 16)
    missing_blocks: threshold value for each column or row to repaint the minority of patches.
    """
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

def mean_img_per_patch(img,block_size):
    '''
    Function that reduces the size of a grayscale image.
    The reduction converts one block or patch to a pixel

    Input:
    img: a grayscale image
    block_size: size of the block used in the segmentation (usually 16)
    '''
    
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    
    numblockwidth = int(imgwidth/block_size)

    numblockheight = int(imgheight/block_size)
    newimg = np.zeros((numblockwidth,numblockheight))
    for i in range(0,numblockwidth):
        for j in range(0, numblockheight):
            pixel_i = i*block_size
            pixel_j = j*block_size
            avg = np.mean(img[pixel_i:pixel_i+block_size,pixel_j:pixel_j+block_size])
            newimg[i,j] = avg
            #img[pixel_i:pixel_i+block_size,pixel_j:pixel_j+block_size] = avg

    return newimg


def pixel_high_change(img, i, j, threshold=4):
    '''
    Function that tells if a given pixel has high change.
    High change means that the number of neighbors with different color 
    is greater than a threshold.

    Input:
    img: a grayscale image
    i,j: indices of the pixel
    threshold: value that delimits if a pixel has high change or not
    
    '''
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    if i > 0 and i < imgwidth - 1 and j > 0 and j < imgheight - 1:
        currcolor = img[i,j]

        #we can compute it
        neighbors = np.zeros(8)

        #black is 0
        neighbors[0] = img[i-1,j]
        neighbors[1] = img[i+1,j]
        neighbors[2] = img[i,j-1]
        neighbors[3] = img[i,j+1]
        neighbors[4] = img[i-1,j-1]
        neighbors[5] = img[i-1,j+1]
        neighbors[6] = img[i+1,j-1]
        neighbors[7] = img[i+1,j+1]
        sumval = np.sum(neighbors)
        if currcolor == 1: #white pixel
            if sumval <= (8-threshold):
                return ("White",True)
            else:
                return ("White",False)
        else: #black pixel
            if sumval >= threshold:
                return ("Black",True)
            else:
                return ("Black",False)

    else:
        #We cannot compute the value for borders, assume borders are always fine, so return 8
        return ("Border", False)