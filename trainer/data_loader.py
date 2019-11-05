import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import PIL
from PIL import Image

import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
IMG_SHAPE = 20

data_dir = "./data"

alphabet = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11,
            "m": 12, "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22,
            "x": 23, "y": 24, "z": 25}


# Creates the dataset.
def load_data():
    file_paths, label_set = load_chars74k_data()

    raw_x = []
    raw_y = []

    for path in file_paths:
        single_x = np.asarray(PIL.Image.open(path)).flatten()
        raw_x.append(single_x)

    for label in label_set:
        transformed_label = char_to_num(label)
        raw_y.append(transformed_label)

    x = np.array(raw_x)
    y = np.array(raw_y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2, train_size=0.8)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], IMG_SHAPE, IMG_SHAPE, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_SHAPE, IMG_SHAPE, 1)

    return x_train, x_test, y_train, y_test


# Needed for transformation of character to number.
def char_to_num(char):
    num = alphabet[char]
    return num


# Needed for transformation of number to character.
def num_to_char(num):
    for key in alphabet:
        if alphabet[key] == num:
            return key


# Load file paths and labels them.
def load_chars74k_data():
    filenames = []
    label_list = []

    for path, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                file = path + "/" + file
                filenames.append(file)

                label = path[-1:]
                label_list.append(label)

    return filenames, label_list


# Use the Keras data generator to augment data.
def create_datagenerator(x_train, x_test, y_train, y_test):
    train_datagen = ImageDataGenerator(
        rotation_range=0. / 180,
        vertical_flip=True)

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(x=x_train, y=y_train)
    validation_generator = test_datagen.flow(x=x_test, y=y_test)

    return train_generator, validation_generator
