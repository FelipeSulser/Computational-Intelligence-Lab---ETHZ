# Computational Intelligence Lab ETHZ 2017

## Road Segmentation

### Preproces
- Histogram equalization failed
- ...

### Feature Selection
- Patch size x-y means that x is the core of the patch and y is the added context
- Label is determinated **only** for the core part
- Patch size: 16-42 works well
- **Even better**: 16-64

### CNN Architecture
- 4 conv-pool layers of depths: 16, 32, 32, 64 and filter sizes: 5, 3, 3, 3
- 3 fully-connected layers of depths: 48, 16, 2
- Outputs softmax

### Post-processing
- Convolution of size 9x9 to smooth
- Binarize the image with threshold 0.5
- 'remove_filtering_neighbors()' with 7 neighboors