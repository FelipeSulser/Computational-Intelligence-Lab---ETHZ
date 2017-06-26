
We have written a simple manual with steps to execute the whole pipeline we used.
Please follow the steps in order so that everything may work.


 1) Put everything together (put the zip file with the code with the dataset together)

 For example, if the zip is unzipped into the folder /user/myuser/cilproject
 then, there needs to be a folder called training with the training dataset at /user/myuser/cilproject/training
 and then another one called 'test_set_images' with the images to predict at /user/myuser/cilproject/test_set_images

 2) Open a terminal and go to /user/myuser/cilproject

 3) Run "python3 increase_dataset.py"

 4) To shuffle the extended datasets run "python3 shuffle_training_set.py"

 5) Train the network opening the file 'tf_aerial_images_deep.py' and set the variable "MODE" to 'train' (on line 55).

 6) Run "python3 tf_aerial_images_deep.py" (will take a long time to train)

 7) To predict, now open the file 'tf_aerial_images_deep.py' and set the variable "MODE" to 'predict' (on line 55).

 8) Run "python3 tf_aerial_images_deep.py" to predict, this will generate images in the folder /user/myuser/cilproject/predictions_test

 9) To train the denoising MLP classifier open the file 'classifier_denoise.py' and set the variable "TRAIN" to 'True' (on line 43).

 10) To speed up the training, you may set n_jobs to a higher number on line 108.

 11) Run "python3 classifier_denoise.py"

 12) To predict, change the variable "TRAIN" to 'False'

 13) Run "python3 classifier_denoise.py"

 14) The images will be written to the folder called /user/myuser/cilproject/predictions_test/result_wavelet

 15) To generate the file to upload to kaggle, run "python3 mask_to_submission.py"

 16) The csv generated is called 'deep_submission.csv' and is located at /user/myuser/cilproject
