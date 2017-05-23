# Computational Intelligence Lab ETHZ 2017

## Road Segmentation

### Data set augmentation
- Rotate each image by 90 deg
- Choose 7 more images with more diagonal roads and highways: [23,26,27,42,72,83,91]
- Rotate these by 180 and 270 deg, so at the end we have 214 input images
- Reshuffle the data set  
- Before balancing the data set, shuffle both classes again


### Preproces
- Histogram equalization failed
- Subtracting mean patch failed


### Feature Selection
- Patch size x-y means that x is the core of the patch and y is the added context
- Label is determinated **only** for the core part
- Patch size: 16-42 works well
- **Even better**: 16-64
- We can try to run PCA on the patches to reduce the dimension from (y,y,3) to (y,y) 


### CNN Architecture

#### Shallow version
- 4 conv-pool layers of depths: 16, 32, 32, 64 and filter sizes: 5, 3, 3, 3
- Max-pooling after each conv. layer with ksize = strides = 2
- 3 fully-connected layers of depths: 48, 16, 2
- Outputs softmax
- **Score obtained**: 0.88629

#### Deep version
- 4 conv-pool layers of depths: 64, 128, 256, 512 and filter sizes: 3, 3, 3, 3
- Max-pooling after each conv. layer with ksize = strides = 2
- 3 fully-connected layers of depths: 2048, 2048, 2
- Outputs softmax
- **Score obtained**: still training


### Post-processing
- Convolution of size 9x9 to smooth
- Binarize the image with threshold 0.5
- `remove_filtering_neighbors()` with 7 neighboors
- Total Variance denoising (TV) was OK, but the convoltuion one was better
- RandomForest with a window of 7x7 patches or 5x5 to predict the center's patch color was OK

