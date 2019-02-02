import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import os
from keras.datasets import mnist
import tensorflow as tf

INPUT_SHAPE=[128,128,3]

BATCH_SIZE=128


channel_num=3



def get_data():
    x=np.load('x_train.npy')
    y=np.load('y_train.npy')
    print(x.shape)
    print(y.shape)
    return(x,y.reshape(100,20))
def train_model(x_train,y_train,x_test,y_test,params,file):
    model = Sequential ()

    model.add (Conv2D (params[0], (3, 3),
                       input_shape=INPUT_SHAPE))
    model.add (Activation ('relu'))
    model.add (Conv2D (params[1], (3, 3)))
    model.add (Activation ('relu'))
    model.add (MaxPooling2D (pool_size=(2, 2)))

    model.add (Conv2D (params[2], (3, 3)))
    model.add (Activation ('relu'))
    model.add (Conv2D (params[3], (3, 3)))
    model.add (Activation ('relu'))
    model.add (MaxPooling2D (pool_size=(2, 2)))

    model.add (Flatten ())
    model.add (Dense (params[4]))
    model.add (Activation ('relu'))
    model.add (Dropout (0.5))
    model.add (Dense (params[5]))
    model.add (Activation ('relu'))
    model.add (Dense (20,activation='softmax'))

    sgd = SGD (lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile (loss=keras.losses.categorical_crossentropy,
                   optimizer=sgd,
                   metrics=['accuracy'])

    checkpoint = keras.callbacks.ModelCheckpoint (file, monitor='val_acc', verbose=1, save_best_only=True,
                                  mode='max')
    callbacks_list = [checkpoint]


    model.fit (x_train, y_train,
               batch_size=128,
               validation_data=(x_test, y_test),
               nb_epoch=50,
               shuffle=True, callbacks=callbacks_list,verbose=0)

x_train,y_train=get_data()

train_model(x_train,y_train,x_train,y_train,[64,64,128,128,200,200],'Asian_model')
