# -*- coding: utf-8 -*-

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.layers import Input

import numpy as np

import scipy.misc
from keras.utils import np_utils
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))

def get_data(data_set, label, img_rows, img_cols):
    X = np.zeros([label.shape[0], img_rows, img_cols])
    for cc, x in enumerate(label.iloc[:, 0]):
        image = data_set + "VirusShare_" + x + ".jpg"
        mat = scipy.misc.imread(image).astype(np.float32)
        X[cc, :, :] = mat
    return X


img_rows, img_cols = 299, 299
nb_classes = 20
nb_epoch = 50

input_shape = (img_rows, img_cols, 1)

label = pd.read_csv('/root/pe_classify/train_label.csv')
x = get_data("/root/pe_classify/pic_299/", label, img_rows, img_cols)
label["Virus_type2"] = pd.factorize(label.iloc[:, 1])[0]
y = label["Virus_type2"]
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
print('x_train shape:', x_train.shape)
print('x_test  shape', x_test.shape)

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
print(y_train.shape)
print(y_test.shape)

input_tensor = Input(shape=(299, 299, 1))

model = InceptionV3(input_tensor=input_tensor, weights=None, include_top=True, classes=nb_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.2, nb_epoch=nb_epoch, shuffle=True, batch_size=8)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


predicted = model.predict(x_test)
y_pred = np.argmax(predicted, 1)
confusion_matrix(y_test, y_pred)
