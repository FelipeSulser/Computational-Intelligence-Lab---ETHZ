import os

import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt



if __name__ == '__main__':
    imgs = []
    data_dir = (os.path.dirname(os.path.realpath(__file__)))+'/training/'
    train_data_input = data_dir + 'images_extended/'
    train_data_output = data_dir + 'images_shuffled/'

    train_labels_input = data_dir + 'groundtruth_extended/' 
    train_labels_output = data_dir + 'groundtruth_shuffled/' 

    if not os.path.isdir(train_data_output):
        os.mkdir(train_data_output)
    if not os.path.isdir(train_labels_output):
        os.mkdir(train_labels_output)            

    save_start_index = 1

    filenames = [f for f in os.listdir(train_data_input) if f.endswith('.png')]
    for filename in filenames:
        prefix = 'satImage_'
        sufix = '.png'
        index = filename[len(prefix):-len(sufix)]
        print(index)
    numfiles = len(filenames)

    input_idx = range(1, numfiles+1)
    input_idx = np.asarray(input_idx)

    np.random.shuffle(input_idx)
    print('Shuffled idx: ', input_idx)


    save_idx = 1

    for i in input_idx:
        imageid = "satImage_%.3d" % i
        image_filename = train_data_input + imageid + ".png"
        label_image_filename = train_labels_input + imageid + ".png"

        if os.path.isfile(image_filename) and os.path.isfile(label_image_filename):
            img = Image.open(image_filename)

            new_imageid = "satImage_%.3d" % save_idx
            img.save(train_data_output+new_imageid+".png")
           
            print ('Loading ' + label_image_filename)
            labelimg = Image.open(label_image_filename)

            label_new_imageid = "satImage_%.3d" % save_idx
            labelimg.save(train_labels_output+label_new_imageid+".png")

            save_idx += 1   

        else:
            print ('File ' + image_filename + ' does not exist')

         
