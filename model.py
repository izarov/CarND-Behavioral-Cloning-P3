import os
import os.path
import pickle
import csv
import numpy as np
import cv2
from keras.models import Model, Sequential, load_model
from keras.layers import Cropping2D, Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
import keras.regularizers as regularizers
from keras.backend import tf as ktf

input_shape = (160, 320, 3)

crop_pixels = ((35, 20), (0, 0)) # crop 52px from the top and 15px from the bottom

cropped_shape = (input_shape[0]-np.sum(crop_pixels[0]), ) + input_shape[1:]

driving_log = 'data/driving_log.csv'


def get_samples():
    """
    Get a list of all samples in the driving log.
    """
    samples = []
    with open(driving_log) as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip header
        for line in reader:
            samples.append(line)

    return samples


def read_img(file_name):
    """
    Read an image from file.
    """
    return cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)


def random_brightness(img):
    """
    Adjust the brightness of an image by a random factor.
    """
    # convert to hsv
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float64)
    # adjust brightness within [-50%,150%]
    random_factor = np.random.uniform(0.5, 1.5)
    img[:,:,2] = random_factor*img[:,:,2]
    # cut off at 255
    img[:,:,2][img[:,:,2]>255]  = 255

    return cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_HSV2RGB)


def flipped_img_processor(left, center, right, angle):
    """
    Given `left`, `center` and `right` camera images paired with an angle
    generate all vertically flipped versions of the images.
    """
    orig = [(left, angle + 0.25), (center, angle), (right, angle - 0.25)]
    flipped = [(cv2.flip(img, 1), -angle) for img, angle in orig]
    return orig + flipped


def generator(samples,
              sample_processor=lambda left, center, right, angle: [(center, angle)],
              batch_size=128):
    """
    Returns a generator of images and angles given a list of samples.

    Loops forever, generating at least `batch_size` number of images at each yield.
    `sample_processor` is a lambda function that returns an array of (image, angle) given lcr input images.
    `batch_size` refers to number of input samples, `sample_processor` may return more than 1 image per sample.
    Applies random_brightness() to each image.
    """
    num_samples = len(samples)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for sample in batch_samples:
                center, left, right = ['data/IMG/' + sample[i].split('/')[-1] for i in range(3)]
                angle = float(sample[3])
                ret_arr = sample_processor(read_img(left), read_img(center), read_img(right), angle)
                for img, angle in ret_arr:
                    img=random_brightness(img)
                    images.append(img)
                    angles.append(angle)

            yield np.array(images), np.array(angles)

        np.random.shuffle(samples)


def nvidia_net(input_shape=input_shape):
    """
    Creates a Keras model with a modified NVIDIA End-to-end CNN architecture.

    Crops top and bottom of image and resizes the result to (66, 200, 3).

    Differences from the original NVIDIA model:

    * ELU activation instead of ReLU (https://arxiv.org/abs/1511.07289)
    * L2 regularization of weights with regularization rate 1e-4
    * Dropout layer after the 5x5 conv filter block
    * Dropout layer after the 3x3 conv filter block
    * Dropout layer after the first Dense layer

    All dropout layers with keep probability of 0.5.

    Original NVIDIA model:
    https://arxiv.org/abs/1604.07316
    """
    model = Sequential()
    model.add(Cropping2D(crop_pixels, input_shape=input_shape))
    model.add(Lambda(lambda x: keras.backend.image.resize_images(x, (66, 200))))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu', W_regularizer=regularizers.l2(1e-4)))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu', W_regularizer=regularizers.l2(1e-4)))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu', W_regularizer=regularizers.l2(1e-4)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, activation='elu', W_regularizer=regularizers.l2(1e-4)))
    model.add(Convolution2D(64, 3, 3, activation='elu', W_regularizer=regularizers.l2(1e-4)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1164, activation='elu', W_regularizer=regularizers.l2(1e-4)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu', W_regularizer=regularizers.l2(1e-4)))
    model.add(Dense(50, activation='elu', W_regularizer=regularizers.l2(1e-4)))
    model.add(Dense(10, activation='elu', W_regularizer=regularizers.l2(1e-4)))
    model.add(Dense(1))

    return model


def train_model(model, X, y, epochs=5):
    """
    Train a model given a static set of X examples with y targets.
    """
    model.compile('adam', 'mse')
    model.fit(X, y, nb_epoch=epochs, validation_split=0.2)
    return model


def train_model_generator(model, samples = None, epochs=5):
    """
    Train a model using a data generator.

    Shuffles samples and reserves 20% for validation.
    """
    if not samples:
        samples = get_samples()

    np.random.shuffle(samples)
    val_start = int(0.8*len(samples))
    train_samples, val_samples = samples[:val_start], samples[val_start:]
    train_gen = generator(train_samples, sample_processor=flipped_img_processor)
    val_gen = generator(train_samples, sample_processor=flipped_img_processor)

    model.compile('adam', 'mse')

    model.fit_generator(train_gen, 2**16, 5,
                        validation_data=val_gen, nb_val_samples=2**14)

    return model


if __name__ == "__main__":
    if not os.path.exists('model.h5'):
        model = train_model_generator(nvidia_net(), epochs=15)
        model.save('model.h5')
