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
#from skimage.restoration import denoise_wavelet
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import utilfuncs


'''
Denoise script that takes as input the raw images outputted by the CNN and performs the following steps:

1)Denoise using wavelets
2)Binarize to b&w
3)Classify border* pixels using MLP

Border pixels are pixels that have more than 5 neighbors (out of 8) with different colors.


You can re-train the classifier by changing Train to True, this will perform a grid search cross validation 
in order to find the best parameters for the MLP.


For more information consult the attached paper (in pdf format) to this script.

'''

TRAIN = True #If false, then predict
PATCH_SIZE = 16
CONTEXT_SIZE = 5 # means that for patch i,j we consider the square i-ps*3,j-ps*3 to i+ps*3, j+ps*3
total_pixel_length = PATCH_SIZE+2*CONTEXT_SIZE*PATCH_SIZE
IMG_SIZE = 608
NEIGHBOOR_TO_CONSIDER = 8


def runscript():
    if TRAIN:
        real_y = []
        train_x = []
        max_val= 0
        train_img_path = (os.path.dirname(os.path.realpath(__file__)))+"/training/groundtruth_extended/"
        label_img_path = (os.path.dirname(os.path.realpath(__file__)))+"/training/groundtruth_extended/"
        for xx in range(1, 310):
            imageid = "satImage_%.3d" % xx
            #imageid = "prediction_"+str(i)
            image_filename = train_img_path +imageid+ ".png"
            img = mpimg.imread(image_filename)
            newimg = utilfuncs.mean_img_per_patch(img,PATCH_SIZE)

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
                    

        train_x = np.asarray(train_x)

        #Scale data
        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)



        real_y = np.asarray(real_y)
        D = train_x.shape[1]
        print("Finished loading images")
        print("Computing model...")
        #rfc = svm.SVC(C=1,kernel='rbf')
        #rfc = RandomForestClassifier(n_estimators=1000,n_jobs=4)
        tuned_parameters = [{'hidden_layer_sizes':[(50,50),(100,),(100,100)], 'activation':['logistic','relu'], 'alpha':[0.01,0.1],'tol':[5e-5,1e-5], 'epsilon': [0.1,0.01,0.001]}]
        #clf = MLPClassifier((10,10),alpha=0.01,epsilon=0.1,tol=1e-4)
        clf = GridSearchCV(MLPClassifier(),tuned_parameters,cv=5,scoring='precision',n_jobs=6)

        #store normalization values to file
        f = open('scale.pckl','wb')
        pickle.dump(scaler,f)
        f.close()

        clf.fit(train_x,real_y) 
        print(clf.best_params_)

        #store MLP model to file
        joblib.dump(clf, 'clfmodel214.pkl') 
    else:
        #We are not training but denoising and classifying

        clf = joblib.load('clfmodel214.pkl')

        #load the normalization values
        f = open('scale.pckl','rb')
        scaler = pickle.load(f)
        f.close()


        #input images to denoise should be located in this directory
        predict_img_path = (os.path.dirname(os.path.realpath(__file__)))+"/predictions_test/result/"

        #output will be written to this directory
        wav_output_img_path = (os.path.dirname(os.path.realpath(__file__)))+"/predictions_test/result_wavelet/"

        #create directory if it does not exist
        if not os.path.isdir(wav_output_img_path):
            os.mkdir(wav_output_img_path) 

        print("Predicting patches")
        for xx in range(1,51):
            imageid = "prediction_"+str(xx)
            image_filename = predict_img_path+imageid+".png"
            print("Predicting "+image_filename)
            img = mpimg.imread(image_filename)
            img = utilfuncs.rgb2gray(img)


            #Inverse colors
            img = 1 - img

            #perform wavelet denoising
            wav_den = denoise_wavelet(img,sigma=3)
            #reverse colors back
            wav_den = 1-wav_den
            
            #Binarize the image to predict on it
            wav_den = utilfuncs.binarize(wav_den,16,0.5)
           
            #now apply the classifier on patches that have high gradient, 
            #This is: > neighbors with different color than their own color
            newimgwav = utilfuncs.mean_img_per_patch(wav_den,PATCH_SIZE)
    
            numblockwidth = newimgwav.shape[0]
            numblockheight = newimgwav.shape[1]
            reswav = []
            for i in range(CONTEXT_SIZE,numblockwidth-CONTEXT_SIZE):
               for j in range(CONTEXT_SIZE,numblockheight-CONTEXT_SIZE):
                    
                    typePatch, isHighChange = utilfuncs.pixel_high_change(newimgwav,i,j,threshold=5)

                    #high change means the patch is a border patch (defined in the paper)
                    if isHighChange:
                        #take its context
                        curr_x = newimgwav[i-CONTEXT_SIZE:i+CONTEXT_SIZE+1,j-CONTEXT_SIZE: j+CONTEXT_SIZE+1]
                        curr_x = curr_x.flatten()
                        full_context = CONTEXT_SIZE*2 +1
                        ind_to_remove = int(((full_context-1)/2)* (1+full_context))
                        #curr_x = np.delete(curr_x,ind_to_remove)
                        reswav.append(curr_x)

            reswav = np.asarray(reswav)

            reswav = scaler.transform(reswav)
            y_estim_wav = clf.predict(reswav)
            it = 0
            for i in range(CONTEXT_SIZE,numblockwidth-CONTEXT_SIZE):
               for j in range(CONTEXT_SIZE,numblockheight-CONTEXT_SIZE):
                typePatch, isHighChange = utilfuncs.pixel_high_change(newimgwav,i,j,threshold=5)
                if isHighChange:
                    #predict on the border patch
                    wav_den[i*PATCH_SIZE:i*PATCH_SIZE+PATCH_SIZE,j*PATCH_SIZE:j*PATCH_SIZE+PATCH_SIZE] = y_estim_wav[it]
                    it+=1

           
            #change the color of a patch if 7 of its neighbors are of a different color
            wav_fil = utilfuncs.remove_filtering_neighbors(wav_den,7,block_size=16)
            #filtered = fill_rows_and_cols(filtered, missing_blocks=3)

            wav_save_str = wav_output_img_path+imageid+".png"
            scipy.misc.imsave(wav_save_str,wav_fil)

if __name__ == '__main__':
    runscript()


