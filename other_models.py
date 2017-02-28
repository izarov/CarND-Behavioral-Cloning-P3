import os
import os.path
import pickle
import csv
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
import keras.applications.vgg16 as vgg16
from keras.models import Model, Sequential, load_model
from keras.layers import Cropping2D, Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
import keras.regularizers as regularizers
from keras.backend import tf as ktf

first_trainable_layer = 15 # keras built-in VGG16 layer idx

def get_samples():
    samples = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip header
        for line in reader:
            samples.append(line)
            
    return samples

def vgg16_preprocessor(img):
    return np.squeeze(vgg16.preprocess_input(np.expand_dims(img, axis=0).astype(np.float64)), axis=0)

def read_img(file_name):
    return cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)

def model_bottom(preprocessor=lambda x: x/255.0 - 0.5, input_shape=input_shape):
    input_tensor = Input(shape=input_shape)
    cropped = Cropping2D(crop_pixels)(input_tensor)
    preprocessed = Lambda(preprocessor)(cropped)
    return Model(input=input_tensor, output=preprocessed)

def model_top():
    model = Sequential()
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1', input_shape=(11, 40, 512)))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='block5_pool'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(2048, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

# BehaveNet is VGG16 with all but the last CNN block and the Dense layers frozen
def behave_net():
    bottom_model = model_bottom()
    # start with the VGG16 model
    base_model = VGG16(include_top=False, input_tensor=bottom_model.output)
    # freeze the first blocks
    for layer in base_model.layers[:first_trainable_layer]:
        layer.trainable = False

    if os.path.exists('model.h5'):
        top = load_model('model.h5')

    model = Model(input_tensor, top(base_model.layers[first_trainable_layer-1].output))
    return model

_bottleneck_split_parts = 10

def extract_bottleneck_features():
    bottom_model = model_bottom()
    # start with the VGG16 model
    base_model = VGG16(include_top=False, input_tensor=bottom_model.output)
    model = Model(input=base_model.input, output=base_model.layers[first_trainable_layer-1].output)
    samples = get_samples()
    images, angles = next(generator(samples, len(samples)-1))
    del samples
    print("{} images and angles".format(len(images), len(angles)))
    pickle.dump(angles, open('angles.p', 'wb'))
    del angles
    bottleneck_output = model.predict(images)
    del images
    # split for serialization to avoid OOM errors
    for i, arr in enumerate(np.array_split(bottleneck_output, _bottleneck_split_parts)):
        pickle.dump(arr, open('bottleneck/{}.p'.format(i), 'wb'), protocol=4)


def load_bottleneck_features():
    assert(os.path.exists('bottleneck/0.p'))
    features = pickle.load(open('bottleneck/0.p', 'rb'))
    for i in range(1, _bottleneck_split_parts):
        features = np.vstack((features, pickle.load(open('bottleneck/{}.p'.format(i), 'rb'))))
    angles = pickle.load(open('angles.p', 'rb'))
    return {'features': features, 'angles': angles}


# https://github.com/commaai/research/blob/master/train_steering_model.py
def commaai_net(input_shape=input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=input_shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    return model
