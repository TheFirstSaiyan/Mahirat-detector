from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#creating the CNN
classifier=Sequential()
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())

#creating the ANN
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

#training the model created
from PIL import Image
classifier.fit_generator(
        training_set,
        steps_per_epoch=1392,
        epochs=25,
        validation_data=test_set,
        validation_steps=280)

file_of_weights="weights_for_player_detector.hdf5"
classifier.save_weights(file_of_weights,overwrite=True)
classifier.load_weights(file_of_weights)

#prediciting for a single picture
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('prediction_set/kohli2.jpg',target_size=(64, 64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
if(result==0):
    print("detected player in the photo : M.S.DHONI")
else:
    print("detected player in the photo : VIRAT KOHLI")
training_set.class_indices

