# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:58:17 2021

@author: wmy
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import load_model
from keras.optimizers import Adam
import random
from keras.layers import Lambda
from optimizer import AdamWithWeightNorm
from utils import DataLoader
from model import AENet

class AEModel(object):
    
    def __init__(self, layers=16):
        self.model = AENet(layers=layers)
        self.model.compile(optimizer=AdamWithWeightNorm(lr=0.001), loss=self.loss)
        pass
    
    def loss(self, y_true, y_pred):
        num_joints = K.shape(y_pred)[-1] // 2
        heatmaps_gt = Lambda(lambda x: x[:, :, :, 0:num_joints])(y_true)
        heatmaps_p = Lambda(lambda x: x[:, :, :, 0:num_joints])(y_pred)
        tapmaps_gt = Lambda(lambda x: x[:, :, :, num_joints:2*num_joints])(y_true)
        tapmaps_p = Lambda(lambda x: x[:, :, :, num_joints:2*num_joints])(y_pred)
        # 回归损失
        heatmaps_loss = mean_squared_error(heatmaps_gt, heatmaps_p)
        batch_size = K.shape(y_true)[0]
        def cond(i, n):
            return tf.less(i, batch_size)
        def body(i, n):
            return tf.add(i, 1), n
        tf.while_loop(cond, body, [0, batch_size])
        return heatmaps_loss
    
    pass