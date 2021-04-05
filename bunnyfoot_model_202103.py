from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential 
from keras.layers import Conv2D ,MaxPooling2D, Flatten, Dense

import numpy as np
import os 


# 모델 생성 및 compile
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 32))
model.add(Dense(units = 1 , activation = 'sigmoid') )

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#model.summary()


model.fit_generator(train_generator,
                         steps_per_epoch = 300,
                         epochs = 5,
                         validation_data = test_generator,
                         validation_steps = 1000
                         )