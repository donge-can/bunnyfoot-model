from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential 
from keras.layers import Conv2D ,MaxPooling2D, Flatten, Dense

import numpy as np
import os 


from google.colab import drive
drive.mount('/content/drive')



### data 불러오기 및 전처리 : n개의 이미지가 폴더에 들어있을 경우
train_path = '/content/drive/Shareddrives/토끼발바닥/train_data'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)




train_generator = train_datagen.flow_from_directory(train_path,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


test_path = '/content/drive/MyDrive/BunnyFoot/'
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(test_path,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


