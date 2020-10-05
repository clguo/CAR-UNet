import os

import numpy as np
import cv2
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, confusion_matrix
from util import *
from keras.callbacks import TensorBoard, ModelCheckpoint
np.random.seed(42)
import scipy.misc as mc
from sklearn.model_selection import KFold
data_location = ''
import  math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


training_images_loc = data_location + 'Stare/X/'
training_label_loc = data_location + 'Stare/Y/'
train_files = os.listdir(training_images_loc)
train_data = []
train_label = []
desired_size=704
for i in train_files:
    im = mc.imread(training_images_loc + i)
    label = mc.imread(training_label_loc + i.split('.')[0] + '.ah.ppm',mode="L")
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    train_data.append(cv2.resize(new_im, (desired_size, desired_size)))
    temp = cv2.resize(new_label,(desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)
train_data = np.array(train_data)

train_label = np.array(train_label)

x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.



TensorBoard(log_dir='./autoencoder', histogram_freq=0,
            write_graph=True, write_images=True)

from sklearn.model_selection import StratifiedKFold
from CARUNet import *

acc_per_fold = []
sen_per_fold = []
pre_per_fold= []
spe_per_fold = []
auc_per_fold =[]
f1_per_fold=[]
iou_per_fold=[]
mcc_per_fold=[]
loss_per_fold = []
fold_no = 1
kfold = KFold(n_splits=4, shuffle=True)
for train, test in kfold.split(x_train, y_train):
    model =CARUNet(input_size=(desired_size, desired_size, 3), start_neurons=16, keep_prob=0.85, block_size=7, lr=1e-3)
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')


    x_rotated, y_rotated, x_flipped, y_flipped= img_augmentation(x_train[train], y_train[train])

    x_train_full = np.concatenate([x_train[train], x_rotated,x_flipped])
    y_train_full = np.concatenate([y_train[train], y_rotated,y_flipped])
    x_train_full, x_validate, y_train_full, y_validate = train_test_split(x_train_full, y_train_full, test_size=0.10, random_state=101)


    x_train_full = np.reshape(x_train_full, (len(x_train_full), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
    y_train_full = np.reshape(y_train_full,(len(y_train_full), desired_size, desired_size, 1))  # adapt this if using `channels_first` im
    x_validate = np.reshape(x_validate, (len(x_validate), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
    y_validate = np.reshape(y_validate,(len(y_validate), desired_size, desired_size, 1))  # adapt this if using `channels_first` im
    history = model.fit(x_train_full, y_train_full,
                        batch_size=3,
                        epochs=80,
                        verbose=0,
                        validation_data=(x_validate, y_validate),
                        # validation_split=0.2,
                        # callbacks=[TensorBoard(log_dir='./autoencoder'), model_checkpoint]
                        )


    y_test = np.reshape( y_train[test], (len(y_train[test]), desired_size, desired_size, 1))  # adapt this if using `channels_first` im
    scores = model.evaluate(x_train[test], y_test, verbose=0)
    y_pred = model.predict(x_train[test])
    y_pred = crop_to_shape(y_pred, (len(y_pred), 605, 700, 1))
    y_pred_threshold = []
    i = 0
    for y in y_pred:
        _, temp = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
        y_pred_threshold.append(temp)
        y = y * 255
        cv2.imwrite('./Stare/result/'+str(fold_no)+'/%d0.85.png' % i, y)
        i += 1
    y_test = crop_to_shape(y_test, (len(y_train[test]), 605, 700, 1))
    y_test = list(np.ravel(y_test))
    y_pred_threshold = list(np.ravel(y_pred_threshold))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

    print('Accuracy:', accuracy_score(y_test, y_pred_threshold))
    print('Sensitivity:', recall_score(y_test, y_pred_threshold))
    print('Specificity', tn / (tn + fp))
    print('NPV', tn / (tn + fn))
    print('PPV', tp / (tp + fp))
    print('AUC:', roc_auc_score(y_test, list(np.ravel(y_pred))))
    print("F1:", 2 * tp / (2 * tp + fn + fp))
    N = tn + tp + fn + fp
    S = (tp + fn) / N
    P = (tp + fp) / N
    print("MCC:", (tp / N - S * P) / math.sqrt(P * S * (1 - S) * (1 - P)))
    print("IOU:", tp / (fp + tp + fn))
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(accuracy_score(y_test, y_pred_threshold))
    sen_per_fold.append(recall_score(y_test, y_pred_threshold))
    spe_per_fold.append(tn / (tn + fp))
    pre_per_fold.append(tp / (tp + fp))
    auc_per_fold.append(roc_auc_score(y_test, list(np.ravel(y_pred))))
    f1_per_fold.append(2 * tp / (2 * tp + fn + fp))
    mcc_per_fold.append((tp / N - S * P) / math.sqrt(P * S * (1 - S) * (1 - P)))
    iou_per_fold.append(tp / (fp + tp + fn))
    loss_per_fold.append(scores[0])
    fold_no = fold_no + 1


print("sen:",np.mean(sen_per_fold))
print("spe:",np.mean(spe_per_fold))
print("acc:",np.mean(acc_per_fold))
print("auc:",np.mean(auc_per_fold))


