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
    return x, ph, pw

def CuttingToOringalShape(x, ph, pw):
    xh = K.shape(x)[1] - ph
    xw = K.shape(x)[2] - pw
    x = Lambda(lambda x: x[:, 0:xh, 0:xw, :])(x)
    return x

def AENet(layers=16, joints=17):
    inputs = Input(shape=(None, None, 3))
    # normalize
    x = Normalization()(inputs)
    # conv2d
    x = Conv2D(32, 5, padding='same')(x)
    # x4
    x4, ph, pw = PaddingToIntegerMultiples(x, integer=4)
    x4 = DesubpixelConv2D(scale=4)(x4)
    x4 = ConvBlock(x4, filters=32, expansion=6, kernel=3)
    # x8
    x8, ph, pw = PaddingToIntegerMultiples(x4, integer=2)
    x8 = DesubpixelConv2D(scale=2)(x8)
    x8 = ConvBlock(x8, filters=64, expansion=6, kernel=3)
    # x16
    x16, ph, pw = PaddingToIntegerMultiples(x8, integer=2)
    x16 = DesubpixelConv2D(scale=2)(x16)
    x16 = ConvBlock(x16, filters=128, expansion=6, kernel=3)
    for i in range(layers):
        # x4
        x4_x4 = ConvBlock(x4, filters=32, expansion=6, kernel=3)
        # upsampling x8
        x8_x4 = ConvBlock(x8, filters=32*2*2, expansion=6, kernel=3)
        x8_x4 = SubpixelConv2D(2)(x8_x4)
        x8_x4 = Lambda(lambda x: x[:, 0:K.shape(x4)[1], 0:K.shape(x4)[2], :])(x8_x4)
        # upsampling x16
        x16_x4 = ConvBlock(x16, filters=32*4*4, expansion=6, kernel=3)
        x16_x4 = SubpixelConv2D(4)(x16_x4)
        x16_x4 = Lambda(lambda x: x[:, 0:K.shape(x4)[1], 0:K.shape(x4)[2], :])(x16_x4)
        # x8
        x8_x8 = ConvBlock(x8, filters=64, expansion=6, kernel=3)
        # upsampling x16
        x16_x8 = ConvBlock(x16, filters=64*2*2, expansion=6, kernel=3)
        x16_x8 = SubpixelConv2D(2)(x16_x8)
        x16_x8 = Lambda(lambda x: x[:, 0:K.shape(x8)[1], 0:K.shape(x8)[2], :])(x16_x8)
        # downsampling x4
        x4_x8, ph, pw = PaddingToIntegerMultiples(x4, integer=2)
        x4_x8 = DesubpixelConv2D(scale=2)(x4_x8)
        x4_x8 = ConvBlock(x4_x8, filters=64, expansion=6, kernel=3)
        # x16
        x16_x16 = ConvBlock(x16, filters=128, expansion=6, kernel=3)
        # downsampling x4
        x4_x16, ph, pw = PaddingToIntegerMultiples(x4, integer=4)
        x4_x16 = DesubpixelConv2D(scale=4)(x4_x16)
        x4_x16 = ConvBlock(x4_x16, filters=128, expansion=6, kernel=3)
        # downsampling x8
        x8_x16, ph, pw = PaddingToIntegerMultiples(x8, integer=2)
        x8_x16 = DesubpixelConv2D(scale=2)(x8_x16)
        x8_x16 = ConvBlock(x8_x16, filters=128, expansion=6, kernel=3)
        # add
        x4 = Add()([x4_x4, x8_x4, x16_x4])
        x8 = Add()([x4_x8, x8_x8, x16_x8])
        x16 = Add()([x4_x16, x8_x16, x16_x16])
        pass
    # x4
    x4_x4 = ConvBlock(x4, filters=32, expansion=6, kernel=3)
    # upsampling x8
    x8_x4 = ConvBlock(x8, filters=32*2*2, expansion=6, kernel=3)
    x8_x4 = SubpixelConv2D(2)(x8_x4)
    x8_x4 = Lambda(lambda x: x[:, 0:K.shape(x4)[1], 0:K.shape(x4)[2], :])(x8_x4)
    # upsampling x16
    x16_x4 = ConvBlock(x16, filters=32*4*4, expansion=6, kernel=3)
    x16_x4 = SubpixelConv2D(4)(x16_x4)
    x16_x4 = Lambda(lambda x: x[:, 0:K.shape(x4)[1], 0:K.shape(x4)[2], :])(x16_x4)
    x4 = Add()([x4_x4, x8_x4, x16_x4])
    # x2 upsampling
    x2 = ConvBlock(x4, filters=joints*2*2*2, expansion=6, kernel=3)
    x2 = SubpixelConv2D(2)(x2)
    xh = K.shape(inputs)[1] // 2
    xw = K.shape(inputs)[2] // 2
    x2 = Lambda(lambda x: x[:, 0:xh, 0:xw, :])(x2)
    # heatmaps
    heatmaps = Lambda(lambda x: x[:, :, :, 0:joints])(x2)
    heatmaps = Activation('sigmoid')(heatmaps)
    # tagmaps
    tagmaps = Lambda(lambda x: x[:, :, :, joints:joints*2])(x2)
    # outputs
    outputs = concatenate([heatmaps, tagmaps], axis=-1)
    return Model(inputs, outputs)


