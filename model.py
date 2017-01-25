# imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.stats import bernoulli
import json
import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

# driving log and image path
DRIVING_LOG_FILE = './data/driving_log.csv'
img_path = './data/'

# Data Augmentation


def crop(image, top_percent, bottom_percent):
    """
    crop the image according to input param
    """
    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]


def resize(image, new_dim):
    """
    using scipy imresize tool, resize the image according to given dimensions
    """
    return scipy.misc.imresize(image, new_dim)


def flip_img(image, steering_angle, flipping_prob=0.5):
    """
    randomly flip an image and take negative of the steering angle for flipped image
    """
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle


def adjust_brightness(image):
    """
    image brightness augmentation
    """
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def img_shear(image, steering_angle, shear_range=200):

    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle


def preprocess_img(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(64, 64), do_shear_prob=0.9):
    """
    crops, resizes and augments the image. The image is resized to 64 x 64 x 3
    to fit in the NVIDIA CNN model.
    """
    head = bernoulli.rvs(do_shear_prob)
    if head == 1:
        image, steering_angle = img_shear(image, steering_angle)

    image = crop(image, top_crop_percent, bottom_crop_percent)

    image, steering_angle = flip_img(image, steering_angle)

    image = adjust_brightness(image)

    image = resize(image, resize_dim)

    return image, steering_angle


def img_angle_file(batch_size=64):
    """
    randomly pick image (left, right or center) and steering angle with
    equal probability. Adjust the steering angle by adding or subtracting
    offset from left and right camera images respectively.
    """
    data = pd.read_csv(DRIVING_LOG_FILE)
    num_of_img = len(data)
    len_random_index = np.random.randint(0, num_of_img, batch_size)
    steering_offset = 0.25

    image_files_and_angles = []
    for index in len_random_index:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + steering_offset
            image_files_and_angles.append((img, angle))

        elif rnd_image == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - steering_offset
            image_files_and_angles.append((img, angle))

    return image_files_and_angles


def generate_next_batch(batch_size=64):
    """
    generator function to yield a training batch with batch_size=64
    """
    while True:
        X_batch = []
        y_batch = []
        images = img_angle_file(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread(img_path + img_file)
            raw_angle = angle
            new_image, new_angle = preprocess_img(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        # assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)


def save_model(model, model_name='model.json', weights_name='model.h5'):
    """
    save the model
    """

    json_string = model.to_json()
    with open(model_name, 'w') as outfile:
        json.dump(json_string, outfile)

    model.save_weights(weights_name)

tf.python.control_flow_ops = tf

epochs = 10
samples_per_epoch = 20032
number_of_validation_samples = 6400
learning_rate = 0.0001
activation = 'relu'

# The CNN model is based on NVIDIA's publication "End to End Learning for Self-Driving Cars"
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

# for the first 3 convolutional layers the kernel_size is 5x5 and stride is 2 x 2.
# for the next 2 convolutional layers the kernel_size is 3x3 and stride is 1x1.
# activation function used is relu.
# the first layer is image normalization layer.

model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation(activation))
model.add(Dropout(0.2))

model.add(Dense(100))
model.add(Activation(activation))
model.add(Dropout(0.2))

model.add(Dense(50))
model.add(Activation(activation))

model.add(Dense(10))
model.add(Activation(activation))

model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(learning_rate), loss="mse", )

# two generators for training and validation
train_gen = generate_next_batch()
validation_gen = generate_next_batch()

history = model.fit_generator(train_gen,
                              samples_per_epoch=samples_per_epoch,
                              nb_epoch=epochs,
                              validation_data=validation_gen,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1)

save_model(model)