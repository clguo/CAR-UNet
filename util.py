

import cv2
import numpy as np

def random_rotation(x_image, y_image):
    rows_x,cols_x, chl_x = x_image.shape
    rows_y,cols_y = y_image.shape
    rand_num = np.random.randint(-40,40)
    M1 = cv2.getRotationMatrix2D((cols_x/2,rows_x/2),rand_num,1)
    M2 = cv2.getRotationMatrix2D((cols_y/2,rows_y/2),rand_num,1)
    x_image = cv2.warpAffine(x_image,M1,(cols_x,rows_x))
    y_image = cv2.warpAffine(y_image,M2,(cols_y,rows_y))
    return np.array(x_image), np.array(y_image)

def h_flip(x_image, y_image):
    x_image = cv2.flip(x_image, 1)
    y_image = cv2.flip(y_image, 1)
    return x_image, y_image

def v_flip(x_image, y_image):
    x_image = cv2.flip(x_image, 0)
    y_image = cv2.flip(y_image, 0)
    return x_image, y_image
def hv_flip(x_image, y_image):
    x_image = cv2.flip(x_image, -1)
    y_image = cv2.flip(y_image, -1)
    return x_image, y_image
def img_augmentation(x_train, y_train):
    x_rotat = []
    y_rotat = []
    x_flip = []
    y_flip = []

    for idx in range(len(x_train)):
        x, y = random_rotation(x_train[idx], y_train[idx])
        x_rotat.append(x)
        y_rotat.append(y)
        x, y = hv_flip(x_train[idx], y_train[idx])
        x_flip.append(x)
        y_flip.append(y)
        x, y = h_flip(x_train[idx], y_train[idx])
        x_flip.append(x)
        y_flip.append(y)
        x, y = v_flip(x_train[idx], y_train[idx])
        x_flip.append(x)
        y_flip.append(y)
    return np.array(x_rotat), np.array(y_rotat), np.array(x_flip), np.array(y_flip)







def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].

    :param data: the array to crop
    :param shape: the target shape
    """
    #

    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    if offset0==0:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:data.shape[1], offset1:(-offset1)]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:data.shape[1], offset1:(-offset1 - 1)]
        else:
            return data[:, offset0:data.shape[1], offset1:(-offset1)]
    elif offset1==0:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:(-offset0 - 1), offset1:data.shape[2]]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:-offset0, offset1:data.shape[2]]
        else:
            return data[:, offset0:-offset0, offset1:data.shape[2]]
    else:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:(-offset0 - 1), offset1:(-offset1)]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:-offset0, offset1:(-offset1-1)]
        else:
            return data[:, offset0:-offset0, offset1:(-offset1)]