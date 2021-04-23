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
from IPython.display import SVG
from keras.utils import plot_model

def SubpixelConv2D(scale, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)

def DesubpixelConv2D(scale, **kwargs):
    return Lambda(lambda x: tf.space_to_depth(x, scale), **kwargs)

def Normalization(**kwargs):
    return Lambda(lambda x: x / 127.5 - 1.0, **kwargs)

def Denormalization(**kwargs):
    return Lambda(lambda x: (x + 1.0) * 127.5, **kwargs)

def ConvBlock(x, filters=64, expansion=6, kernel=3):
    x = Conv2D(filters * expansion, 1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(int(filters * 0.8), 1, padding='same')(x)
    x = Conv2D(filters, kernel, padding='same')(x)
    return x

def PaddingToIntegerMultiples(x, integer):
    ph = (integer - K.shape(x)[1] % integer) % integer
    pw = (integer - K.shape(x)[2] % integer) % integer
    x = Lambda(tf.pad, arguments={'paddings':[[0, 0], [0, ph], [0, pw], [0, 0]]})(x)
    return x, ph, pw

def CuttingToOringalShape(x, ph, pw):
    xh = K.shape(x)[1] - ph
    xw = K.shape(x)[2] - pw
    x = Lambda(lambda x: x[:, 0:xh, 0:xw, :])(x)
    return x

def AEModel(filters=32, blocks=32):
    inputs = Input(shape=(None, None, 3))
    # normalize
    x = Normalization()(inputs)
    # padding 
    x, ph, pw = PaddingToIntegerMultiples(x, integer=4)
    # space to depth
    outputs = DesubpixelConv2D(scale=4)(x)
    return Model(inputs, outputs)