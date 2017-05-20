# Computational-Intelligence-Lab---ETHZ

## CIL project 2017

### Road Segmentation
- Preproces: None
- Patch size: 16-42 works well, we are now trying 16-64
- CNN architecture (TBD)
- Postprocessing: 
-- convolution of size 9x9 to smooth
-- binarize
-- remove_filtering_neighbors() with 7 neighboors
