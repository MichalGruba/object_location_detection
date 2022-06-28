import tensorflow as tf
import os
import sys
import glob
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import random
from model import getModel
import prep

# paths
home_path = os.getcwd()
tool_path = home_path + "/tool/"
imagesPath = home_path + "/images/"
ds_path = home_path + "/ds"
ds_masks_path = ds_path + "/masks"
ds_img_path = ds_path + "/images"

try:
    os.mkdir(ds_path)
    os.mkdir(ds_masks_path)
    os.mkdir(ds_img_path)
except FileExistsError:
    pass

IMG_SIZE = 224
a = sys.argv[1:]
if not a:
    a.append(tool_path)
img = [cv2.imread(item) for i in [glob.glob(str(a[0]) + '*.%s' % ext) for ext in ["jpg", "png"]] for item in i]


# creates a mask using min and max threshold values
def get_mask(im, th_min, th_max):
    dst = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    _, dst = cv2.threshold(dst, th_min, th_max, cv2.THRESH_BINARY_INV)
    return dst


# gets bounding box
def get_bounding_box(image):
    cont, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_lst = [(cv2.contourArea(cnt), cnt) for cnt in cont]
    cont_lst = sorted(cont_lst, key=lambda c: c[0], reverse=True)
    p_x, p_y, w, h = cv2.boundingRect(cont_lst[0][1])
    return p_x, p_y, w, h


# creates dataset by inserting randomly rotated tools into pictures
def create_data_set():
    img_list = [cv2.imread(item) for i in [glob.glob(str(imagesPath) + '*.%s' % ext) for ext in ["jpg", "png"]] for item
                in i]
    idx = 1
    label_list = []
    for i in range(200):
        # pick random image and resize it
        im_num = random.randint(0, img_list.__len__() - 1)
        im = cv2.resize(img_list[im_num], (2000, 2000), interpolation=cv2.INTER_AREA)

        # pick a random tool and angle
        tool_num = random.randint(0, tools.__len__() - 1)
        random_angle = random.uniform(0, 360)

        # rotate the tool and pick a random position
        tool = imutils.rotate_bound(tools[tool_num], random_angle)
        pos_x = random.randint(0, im.shape[0] - tool.shape[0] - 1)
        pos_y = random.randint(0, im.shape[1] - tool.shape[1] - 1)

        # create a mask to paste a tool
        tool_mask = get_mask(tool, 1, 255)
        tool_mask = tool_mask.reshape(tool_mask.shape[0], tool_mask.shape[1], 1)

        # take a fragment of a picture and paste the tool there
        frag = im[pos_x:pos_x + tool.shape[0], pos_y:pos_y + tool.shape[1]]
        frag = np.where(tool_mask != 0, frag, tool)
        im[pos_x:pos_x + tool.shape[0], pos_y:pos_y + tool.shape[1]] = frag

        # create an empty image and paste the tool mask there
        empty_img = np.zeros((im.shape[0], im.shape[1], 1))
        empty_frag = empty_img[pos_x:pos_x + tool.shape[0], pos_y:pos_y + tool.shape[1]]
        empty_img[pos_x:pos_x + tool.shape[0], pos_y:pos_y + tool.shape[1]] = np.where(tool_mask != 0, empty_frag, 1)

        # resize the images
        empty_img = cv2.resize(empty_img, (IMG_SIZE, IMG_SIZE))
        im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))

        # get rid of interpolation
        empty_img = np.where(empty_img > 0.5, 1, 0)

        # get bounding box of a tool
        empty_img = (empty_img * 255).astype(np.uint8)
        x, y, w, h = get_bounding_box(empty_img)
        b_box = x / empty_img.shape[0], y / empty_img.shape[1], (x + w) / empty_img.shape[0], (y + h) / empty_img.shape[
            1]

        # save the dataset
        label_list.append(b_box)
        os.chdir(ds_masks_path)
        cv2.imwrite(str(idx) + ".tif", empty_img)
        os.chdir(ds_img_path)
        cv2.imwrite(str(idx) + ".tif", im)

        idx = idx + 1
    os.chdir(home_path)
    np.savetxt("labels.csv", label_list, delimiter=",")


tools = []
for i in img:
    # remove background and create mask
    masks = []
    mask = get_mask(i, 240, 255)
    i = cv2.bitwise_and(i, i, mask=mask)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ellipse and get the angle of rotation
    ellipse = cv2.fitEllipse(contours[-1])
    angle = ellipse[2]

    # rotate image and mask
    rot = imutils.rotate_bound(i, -angle)
    mask_rot = imutils.rotate_bound(mask, -angle)

    # get moments and make sure the tool is facing upwards
    moments = cv2.moments(mask)
    m_x = int(moments["m10"] / moments["m00"])
    m_y = int(moments["m01"] / moments["m00"])
    if m_y < i.shape[1] // 2:
        rot = cv2.flip(rot, 0)
        mask_rot = cv2.flip(mask_rot, 0)

    # find contours and create bounding box
    X, Y, W, H = get_bounding_box(mask_rot)

    # crop image
    rot = rot[Y:Y + H, X:X + W]

    tools.append(rot)


create_data_set()

# load data
labels = np.genfromtxt(home_path + "/labels.csv", delimiter=',')
images_path = ds_img_path
images = prep.get_images(images_path)

# normalize the images
images = prep.normalize(images)

# shuffle data
Data = prep.shuffle_data(images, labels)

# split data into training and testing set
training_data = Data[:int(0.9 * len(Data))]
testing_data = Data[int(0.1 * len(Data)):]

# split data into images and labels
X, Y = prep.split_data(training_data)
X_test, y_test = prep.split_data(testing_data)

# create and compile model
homePath = home_path + "model"
if not (os.path.isfile(homePath + ".h5") and os.path.isfile(homePath + ".json")):
    model = getModel(X)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # start training and display summary
    model.fit(X, Y, batch_size=10, epochs=30, validation_split=0.1)
    model.summary()

    # save the model
    os.chdir(home_path)
    model_json = model.to_json()
    with open("model.json", "w") as model_file:
        model_file.write(model_json)
    model.save_weights("model.h5")
else:
    # load the model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# evaluate model on a test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2, batch_size=10)

# predict and display
for i in range(9):
    prediction = model.predict(X_test[i].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

    # get the predicted bounding box corners
    X1, Y1, X2, Y2 = prediction[0][0], prediction[0][1], prediction[0][2], prediction[0][3]
    X1, Y1, X2, Y2 = int(X1 * X_test[i].shape[0]), int(Y1 * X_test[i].shape[1]), \
        int(X2 * X_test[i].shape[0]), int(Y2 * X_test[i].shape[1])

    # draw it
    cv2.rectangle(X_test[i], (X1, Y1), (X2, Y2),
                  (0, 255, 0), 2)

    # get the real bounding box corners
    X1t, Y1t, X2t, Y2t = int(y_test[i][0] * X_test[i].shape[0]), int(y_test[i][1] * X_test[i].shape[1]), \
        int(y_test[i][2] * X_test[i].shape[0]), int(y_test[i][3] * X_test[i].shape[1])

    # draw it
    cv2.rectangle(X_test[i], (X1t, Y1t), (X2t, Y2t),
                  (0, 0, 255), 2)

    # plot the images
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i])

plt.show()

# check the external images
ext_path = home_path + "/ext/"

ims = [cv2.imread(item) for i in [glob.glob(str(ext_path) + '*.%s' % ext) for ext in ["jpg", "png"]] for item
       in i]
if len(ims) > 0:
    X_test_ext = prep.resize_images(ims)
    X_test_ext = prep.normalize(X_test_ext)
    test = np.array(X_test_ext)

    n = 0
    for i in range(4):
        prediction = model.predict(test[i].reshape(-1, IMG_SIZE, IMG_SIZE, 3))
        X1, Y1, X2, Y2 = prediction[0][0], prediction[0][1], prediction[0][2], prediction[0][3]
        X1, Y1, X2, Y2 = int(X1 * test[i].shape[0]), int(Y1 * test[i].shape[1]), int(X2 * test[i].shape[0]), int(
            Y2 * test[i].shape[1])
        cv2.rectangle(test[i], (X1, Y1), (X2, Y2),
                      (0, 255, 0), 2)
        plt.subplot(3, 3, n + 1)
        plt.imshow(test[i])
        n = n + 1

    plt.show()
