import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import scipy.misc
from sklearn import svm
import math
import pickle
import os
from sklearn.externals import joblib
from skimage import color, data, restoration
from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_wavelet
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

TRAIN = False #If false, then predict
PATCH_SIZE = 16
CONTEXT_SIZE = 5 # means that for patch i,j we consider the square i-ps*3,j-ps*3 to i+ps*3, j+ps*3
# Create graph
total_pixel_length = PATCH_SIZE+2*CONTEXT_SIZE*PATCH_SIZE
#sess = tf.Session()
IMG_SIZE = 608
NEIGHBOOR_TO_CONSIDER = 8


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def binarize(img,block_size,threshold):
    
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

def remove_filtering_neighbors(img,black_threshold, block_size = 16):
    #img is b&w array with 0 or 1
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

def runscript():
    if TRAIN:
        real_y = []
        train_x = []
        max_val= 0
        train_img_path = (os.path.dirname(os.path.realpath(__file__)))+"/training/groundtruth_extended/"
        label_img_path = (os.path.dirname(os.path.realpath(__file__)))+"/training/groundtruth_extended/"
        for xx in range(1, 312):
            imageid = "satImage_%.3d" % xx
            #imageid = "prediction_"+str(i)
            image_filename = train_img_path +imageid+ ".png"
            img = mpimg.imread(image_filename)
            newimg = mean_img_per_patch(img,PATCH_SIZE)

            numblockwidth = newimg.shape[0]

            numblockheight = newimg.shape[1]
            #print(img)
            #2D matrix between 0,1
            num_patches = int(img.shape[0]/PATCH_SIZE)
            for i in range(CONTEXT_SIZE,numblockwidth-CONTEXT_SIZE):
                for j in range(CONTEXT_SIZE,numblockheight-CONTEXT_SIZE):
                    curr_x = newimg[i-CONTEXT_SIZE:i+CONTEXT_SIZE+1,j-CONTEXT_SIZE: j+CONTEXT_SIZE+1]
                    curr_x = curr_x.flatten()
                    full_context = CONTEXT_SIZE*2 +1
                    ind_to_remove = int(((full_context-1)/2)* (1+full_context))
                    #curr_x = np.delete(curr_x,ind_to_remove)
                    #img[(i*PATCH_SIZE - CONTEXT_SIZE*PATCH_SIZE):(i*PATCH_SIZE+CONTEXT_SIZE*PATCH_SIZE),(j*PATCH_SIZE-CONTEXT_SIZE*PATCH_SIZE):(j*PATCH_SIZE+CONTEXT_SIZE*PATCH_SIZE)]
                    curr_patch = newimg[i,j]
                    mean_val = np.mean(curr_patch)
                    if mean_val > 0.25:
                        real_y.append(1)
                    else:
                        real_y.append(0)


                    flattened = curr_x.flatten()
                    train_x.append(flattened)
                    

        #BEST PARAMETERS SVC C=10, gamma = 0.1
        train_x = np.asarray(train_x)
        real_y = np.asarray(real_y)
        D = train_x.shape[1]
        print("Finished loading images")
        print("Computing model...")
        #rfc = svm.SVC(C=1,kernel='rbf')
        #rfc = RandomForestClassifier(n_estimators=1000,n_jobs=4)

        tuned_parameters = [{'hidden_layer_sizes':[(50,50),(100,),(100,100)], 'activation':['logistic','relu'], 'alpha':[0.01,0.1],'tol':[1e-4,1e-5]}]
        #clf = MLPClassifier((10,10),alpha=0.01,epsilon=0.1,tol=1e-4)
        clf = GridSearchCV(MLPClassifier(),tuned_parameters,cv=5,scoring='precision')

        # gammas = np.array([0.1,0.01,0.001])
        # cs = np.array([10,1,0.1])
        # grid_dict = dict(gamma=gammas,C=cs)
        # classifier = GridSearchCV(estimator = clf,param_grid=grid_dict,n_jobs=4,cv=5)
        # classifier.fit(train_x,real_y)
        # print("BEST PARAMETERS")
        # print(classifier.best_params_)
        clf.fit(train_x,real_y) 
        print(clf.get_params())
        joblib.dump(rfc, 'clfmodel214.pkl') 
    else:
        clf = joblib.load('clfmodel214.pkl')
        #now predict

        predict_img_path = (os.path.dirname(os.path.realpath(__file__)))+"/predictions_test/result_azure_residual/result/"


        tv_output_img_path = (os.path.dirname(os.path.realpath(__file__)))+"/predictions_test/result_tv/"
        wav_output_img_path = (os.path.dirname(os.path.realpath(__file__)))+"/predictions_test/result_wavelet/"
        if not os.path.isdir(tv_output_img_path):
            os.mkdir(tv_output_img_path) 

        if not os.path.isdir(wav_output_img_path):
            os.mkdir(wav_output_img_path) 

        print("Predicting patches")
        for xx in range(1,51):
            imageid = "prediction_"+str(xx)
            image_filename = predict_img_path+imageid+".png"
            print("Predicting "+image_filename)
            img = mpimg.imread(image_filename)
            img = rgb2gray(img)


            #tv denoise works best with inverse colors
            img = 1 - img
            tv_denoise = denoise_tv_chambolle(img, weight=10)
            wav_den = denoise_wavelet(img,sigma=3)
            wav_den = 1-wav_den
            wav_den = binarize(wav_den,16,0.5)
            tv_denoise = 1 - tv_denoise
            tv_denoise_bw = binarize(tv_denoise,16,0.5)
            
           
            #now apply the classifier on patches that have high gradient, 
            #This is: >= neighbors with different color than their own color
            newimgwav = mean_img_per_patch(wav_den,PATCH_SIZE)
            newimg = mean_img_per_patch(tv_denoise_bw,PATCH_SIZE)
            numblockwidth = newimg.shape[0]
            numblockheight = newimg.shape[1]
            reswav = []
            for i in range(CONTEXT_SIZE,numblockwidth-CONTEXT_SIZE):
               for j in range(CONTEXT_SIZE,numblockheight-CONTEXT_SIZE):
                    
                    typePatch, isHighChange = pixel_high_change(newimgwav,i,j,threshold=5)
                    if isHighChange:
                        curr_x = newimgwav[i-CONTEXT_SIZE:i+CONTEXT_SIZE+1,j-CONTEXT_SIZE: j+CONTEXT_SIZE+1]
                        curr_x = curr_x.flatten()
                        full_context = CONTEXT_SIZE*2 +1
                        ind_to_remove = int(((full_context-1)/2)* (1+full_context))
                        #curr_x = np.delete(curr_x,ind_to_remove)
                        reswav.append(curr_x)

            reswav = np.asarray(reswav)
            y_estim_wav = clf.predict(reswav)
            it = 0
            for i in range(CONTEXT_SIZE,numblockwidth-CONTEXT_SIZE):
               for j in range(CONTEXT_SIZE,numblockheight-CONTEXT_SIZE):
                typePatch, isHighChange = pixel_high_change(newimgwav,i,j,threshold=5)
                if isHighChange:
                    wav_den[i*PATCH_SIZE:i*PATCH_SIZE+PATCH_SIZE,j*PATCH_SIZE:j*PATCH_SIZE+PATCH_SIZE] = y_estim_wav[it]
                    it+=1

           
            
           
            res = []
            for i in range(CONTEXT_SIZE,numblockwidth-CONTEXT_SIZE):
               for j in range(CONTEXT_SIZE,numblockheight-CONTEXT_SIZE):
                    
                    typePatch, isHighChange = pixel_high_change(newimg,i,j,threshold=5)
                    if isHighChange:
                        curr_x = newimg[i-CONTEXT_SIZE:i+CONTEXT_SIZE+1,j-CONTEXT_SIZE: j+CONTEXT_SIZE+1]
                        curr_x = curr_x.flatten()
                        full_context = CONTEXT_SIZE*2 +1
                        ind_to_remove = int(((full_context-1)/2)* (1+full_context))
                        #curr_x = np.delete(curr_x,ind_to_remove)
                        res.append(curr_x)

            res = np.asarray(res)
            y_estim = clf.predict(res)
            it = 0
            for i in range(CONTEXT_SIZE,numblockwidth-CONTEXT_SIZE):
               for j in range(CONTEXT_SIZE,numblockheight-CONTEXT_SIZE):
                typePatch, isHighChange = pixel_high_change(newimg,i,j,threshold=5)
                if isHighChange:
                    tv_denoise_bw[i*PATCH_SIZE:i*PATCH_SIZE+PATCH_SIZE,j*PATCH_SIZE:j*PATCH_SIZE+PATCH_SIZE] = y_estim[it]
                    it+=1

            filtered = remove_filtering_neighbors(tv_denoise_bw,7,block_size=16)

            wav_fil = remove_filtering_neighbors(wav_den,7,block_size=16)
            #filtered = fill_rows_and_cols(filtered, missing_blocks=3)

            save_str = tv_output_img_path+imageid+".png"
            scipy.misc.imsave(save_str,filtered)

            wav_save_str = wav_output_img_path+imageid+".png"
            scipy.misc.imsave(wav_save_str,tv_denoise)

if __name__ == '__main__':
    runscript()


