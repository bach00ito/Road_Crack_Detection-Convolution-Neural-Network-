import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
train_set=ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2)
train=train_set.flow_from_directory('dataset2/train',target_size=(64,64),color_mode='rgb',class_mode='binary',batch_size=200)

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=30,activation='relu',kernel_initializer='he_uniform'))
cnn.add(tf.keras.layers.Dense(units=20,activation='relu',kernel_initializer='he_uniform'))
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform')) 

cnn.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
cnn.fit(x = train, epochs = 3) 

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset2/single_prediction/crack.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)

if result[0][0] == 1:
    prediction = 'crack'
else:
    prediction = 'no crack'
print(prediction)
