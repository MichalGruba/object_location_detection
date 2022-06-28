import cv2
import numpy as np
import glob
import random

IMG_SIZE = 224


# gets an array of images
def get_images(dir_path):
    images = [cv2.imread(item) for i in [glob.glob(str(dir_path) + '/%i.tif' % num) for num in range(200)] for
              item in i]
    images = [cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in images]
    images = np.array(images).reshape((-1, IMG_SIZE, IMG_SIZE, 3))
    return images


# normalizes images' pixel values
def normalize(images):
    temp1 = []
    for i in range(images.__len__()):
        temp1.append(images[i].astype(np.float32))
        temp1[i] = temp1[i] / 255.0
    return temp1


# shuffles the data
def shuffle_data(images, labels):
    data = list(zip(images, labels))
    random.shuffle(data)
    return data


# splits list of tuples (img, label) into two separate lists
def split_data(data):
    images = []
    labels = []
    for img, label in data:
        images.append(img)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# prepares a single picture to feed it into the model
def prepare(filepath):
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = (new_array / 255.0).astype(np.float32)
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


# resizes the images
def resize_images(images):
    ims = []
    for im in images:
        tmp = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
        ims.append(tmp)
    return ims

