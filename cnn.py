# Convolutional Neural Network for image classification

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras




# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt


#Contants
PIXEL = 64 #Input images are read in this pixel. Here it means 64*64. Set this to higher values for better results.
EPOCH = 2 #No. of times the model is being trained through all the images. 
TRAIN_IMAGE_NUM = 800 #No. of images in training set
TEST_IMAGE_NUM = 200 #No. of images in test set
PREDICT_IMAGE_NUM = 10 #No. images to predict, containing in predict_set folder
FEATURE_DET = 32

# Recommended constants, At PIXEL 128, EPOCH 20, FEATURE_DET = 64, I GOT 90% ACCURACY, but It takes almost 2 hours to run !


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(FEATURE_DET, 3, 3, input_shape = (PIXEL, PIXEL, 3), activation = 'relu')) # relu means applying rectifier function
#we have set 64 feature detector. The input images are forced to be read as 128*128. 
#These 3 are rgb colors dimension 

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#pooling reduces the dimension of convolution layer.

# Adding a second convolutional layer for improving accuracy of the model.
classifier.add(Convolution2D(FEATURE_DET, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening : converts the pooling matrix into single long vectors by preserving the features.
classifier.add(Flatten())

# Step 4 - Full connection. From here on is normal artificial neural network
classifier.add(Dense(output_dim = 128, activation = 'relu')) #128, chose this number by experamenting 
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])






# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
#Following fuction rescales, zoom in, flips our training and test images so that it improves accuracy
# of the model by enriching different aspects of images.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('ships/training_set',
                                                 target_size = (PIXEL, PIXEL),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('ships/test_set',
                                            target_size = (PIXEL, PIXEL),
                                            batch_size = 32,
                                            class_mode = 'binary')
#Training the model 
classifier.fit_generator(training_set,
                         samples_per_epoch = TRAIN_IMAGE_NUM,
                         nb_epoch = EPOCH,
                         validation_data = test_set,
                         nb_val_samples = TEST_IMAGE_NUM)





# Part 3 - Saving the classifier model for future prediction
#label_map contains the mapping types and it's indicating value. i.e 0 -> container, 1 -> tanker 
label_map = training_set.class_indices

classifier.save('ship_img_cla_model.h5')


loaded_classifier = tf.keras.models.load_model('ship_img_cla_model.h5')
#loaded_classifier.layers[0].input_shape #(None, 128, 128, 3)

#iterating through the images in predict_set folder and saving the image pixels in 128*128 in batch_holder  
batch_holder = np.zeros((PREDICT_IMAGE_NUM, PIXEL, PIXEL, 3))
img_dir='predict_set/'
for i,img in enumerate(os.listdir(img_dir)):
  img = image.load_img(os.path.join(img_dir,img), target_size=(PIXEL,PIXEL))
  batch_holder[i, :] = img

#Predicting 
result=loaded_classifier.predict_classes(batch_holder)
result_classes = result.argmax(axis=-1)

#Visualization
fig = plt.figure(figsize=(20, 20))
 
for i,img in enumerate(batch_holder):
  fig.add_subplot(4,5, i+1)
  plt.title(result_classes[i])
  plt.imshow(img/256.)
  
plt.show()

