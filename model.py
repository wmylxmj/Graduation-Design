# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:58:31 2021
@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.layers import Add, Conv2D, Input, Lambda, Activation, Conv2DTranspose
from keras.models import Model
from keras.layers import Conv3D, ZeroPadding3D, BatchNormalization, Multiply
from keras.layers import LeakyReLU, concatenate, Reshape, Softmax, MaxPool2D
from keras.layers import Dense
from keras.layers import Conv1D, ZeroPadding1D, AveragePooling1D, Flatten
from keras.layers import Dropout
from IPython.display import SVG
from keras.utils import plot_model
from tensorflow.python import keras


'''SPENet'''

def SubpixelConv2D(scale, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)

def DesubpixelConv2D(scale, **kwargs):
    return Lambda(lambda x: tf.space_to_depth(x, scale), **kwargs)

def Normalization(**kwargs):
    return Lambda(lambda x: x / 127.5 - 1.0, **kwargs)

def Denormalization(**kwargs):
    return Lambda(lambda x: (x + 1.0) * 127.5, **kwargs)

def ConvBlock(x, filters=32, expansion=6, kernel=3):
    x = Conv2D(filters * expansion, 1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(int(filters * 0.8), 1, padding='same')(x)
    x = Conv2D(filters, kernel, padding='same')(x)
    return x

def PaddingToIntegerMultiples(x, integer):
    ph = (integer - K.shape(x)[1] % integer) % integer
    pw = (integer - K.shape(x)[2] % integer) % integer
    x = Lambda(tf.pad, arguments={'paddings':[[0, 0], [0, ph], [0, pw], [0, 0]]})(x)
    return x

def UpSamplingBlock(x, filters, scale):
    x_y = ConvBlock(x, filters=filters*scale**2, expansion=6, kernel=3)
    x_y = SubpixelConv2D(scale)(x_y)
    return x_y

def DownSamplingBlock(x, filters, scale):
    x_y = DesubpixelConv2D(scale=scale)(x)
    x_y = ConvBlock(x_y, filters=filters, expansion=6, kernel=3)
    return x_y

def SPENet(layers=8, joints=17):
    inputs = Input(shape=(None, None, 3))
    # normalize
    x = Normalization()(inputs)
    # padding
    x = PaddingToIntegerMultiples(x, integer=16)
    # conv2d
    x = Conv2D(32, 5, padding='same')(x)
    # x4
    x4 = DownSamplingBlock(x, filters=32, scale=4)
    # x8
    x8 = DownSamplingBlock(x4, filters=64, scale=2)
    # x16
    x16 = DownSamplingBlock(x8, filters=128, scale=2)
    for i in range(layers):
        # x4
        x4_x4 = ConvBlock(x4, filters=32, expansion=6, kernel=3)
        # upsampling x8
        x8_x4 = UpSamplingBlock(x8, filters=32, scale=2)
        # upsampling x16
        x16_x4 = UpSamplingBlock(x16, filters=32, scale=4)
        # x8
        x8_x8 = ConvBlock(x8, filters=64, expansion=6, kernel=3)
        # upsampling x16
        x16_x8 = UpSamplingBlock(x16, filters=64, scale=2)
        # downsampling x4
        x4_x8 = DownSamplingBlock(x4, filters=64, scale=2)
        # x16
        x16_x16 = ConvBlock(x16, filters=128, expansion=6, kernel=3)
        # downsampling x4
        x4_x16 = DownSamplingBlock(x4, filters=128, scale=4)
        # downsampling x8
        x8_x16 = DownSamplingBlock(x8, filters=128, scale=2)
        # add
        x4 = Add()([x4, x4_x4, x8_x4, x16_x4])
        x8 = Add()([x8, x4_x8, x8_x8, x16_x8])
        x16 = Add()([x16, x4_x16, x8_x16, x16_x16])
        pass
    # x4
    x4_x4 = ConvBlock(x4, filters=32, expansion=6, kernel=3)
    # upsampling x8
    x8_x4 = UpSamplingBlock(x8, filters=32, scale=2)
    # upsampling x16
    x16_x4 = UpSamplingBlock(x16, filters=32, scale=4)
    x4 = Add()([x4, x4_x4, x8_x4, x16_x4])
    # x2 upsampling
    x2 = ConvBlock(x4, filters=joints*2**2, expansion=6, kernel=3)
    x2 = SubpixelConv2D(2)(x2)
    # heatmaps
    xh = K.shape(inputs)[1] // 2
    xw = K.shape(inputs)[2] // 2
    outputs = Lambda(lambda x: x[:, 0:xh, 0:xw, :])(x2)
    return Model(inputs, outputs)

'''TCPNet'''

def DenseBlock(X, nF, nG, nD):
    for i in range(nD):
        T = BatchNormalization(axis=2)(X)
        T = LeakyReLU(alpha=0.1)(T)
        T = Conv1D(filters=nF, kernel_size=1, strides=1, padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(T)
        T = BatchNormalization(axis=2)(T)
        T = LeakyReLU(alpha=0.1)(T)
        T = ZeroPadding1D(padding=1)(T)
        T = Conv1D(filters=nG, kernel_size=3, strides=1, padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(T)
        X = concatenate([X, T], axis=2)
        nF += nG
        pass
    return X

def ResidualDenseBlock(X, nC_in, nC_out, nF, nG, nD, strides=1):
    branch = DenseBlock(X, nF, nG, nD)
    branch = Conv1D(filters=nC_out, kernel_size=1, strides=strides, padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(branch)
    if nC_in != nC_out or strides != 1:
        X = Conv1D(filters=nC_out, kernel_size=1, strides=strides, padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(X)
        pass
    X = Add()([branch, X])
    return X

def TCPNet(step=136, features=51, ndim=6):
    X_in = Input((step, features))
    X = ZeroPadding1D(padding=3)(X_in)
    X = Conv1D(filters=32, kernel_size=7, strides=2, padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(X)
    X = ResidualDenseBlock(X, 32, 32, 16, 4, 4, strides=2)
    X = ResidualDenseBlock(X, 32, 64, 32, 8, 4, strides=2)
    X = ResidualDenseBlock(X, 64, 128, 64, 16, 4, strides=2)
    X = ResidualDenseBlock(X, 128, 256, 128, 32, 4, strides=2)
    X = AveragePooling1D(pool_size=2)(X)
    X = Flatten()(X)
    X = Dense(ndim, activation='softmax')(X)
    model = Model(X_in, X)
    return model