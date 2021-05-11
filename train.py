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
        # 关节数 17
        num_joints = K.shape(y_pred)[-1] // 2
        # heatmaps
        heatmaps_gt = Lambda(lambda x: x[:, :, :, 0:num_joints])(y_true)
        heatmaps_p = Lambda(lambda x: x[:, :, :, 0:num_joints])(y_pred)
        # tagmaps
        tagmaps_gt = Lambda(lambda x: x[:, :, :, num_joints:2*num_joints])(y_true)
        tagmaps_gt = tf.cast(tagmaps_gt, dtype=tf.int32)
        tagmaps_p = Lambda(lambda x: x[:, :, :, num_joints:2*num_joints])(y_pred)
        # 回归损失
        heatmaps_loss = mean_squared_error(heatmaps_gt, heatmaps_p)
        # 类别损失
        batch_size = K.shape(y_true)[0]
        # loop batches
        def cond_b(batch, batch_size, tagmaps_gt, tagmaps_p):
            return tf.less(batch, batch_size)
        def body_b(batch, batch_size, tagmaps_gt, tagmaps_p):
            tagmaps_gt_single = Lambda(lambda x: x[batch, :, :, :])(tagmaps_gt)
            tagmaps_p_single = Lambda(lambda x: x[batch, :, :, :])(tagmaps_p)        
            n_people = tf.cast(tf.reduce_max(tagmaps_gt_single), dtype=tf.int32)
            # loop peoples
            def cond_p(people, n_people, tagmaps_gt_single, tagmaps_p_single):
                return tf.less(people, n_people)
            def body_p(people, n_people, tagmaps_gt_single, tagmaps_p_single):
                joints = tf.where(tf.equal(tagmaps_gt_single, people+1))
                n_joints = K.shape(joints)[0] 
                hn_hat = 0.0
                # loop joints
                def cond_j(joint, n_joints, joints, hn_hat):
                    return tf.less(joint, n_joints)
                def body_j(joint, n_joints, joints, hn_hat):
                    x, y, j = joints[joint][0], joints[joint][1], joints[joint][2]
                    # 第n个人第k个关节的tag值
                    hk_xnk = tagmaps_p_single[x][y][j]
                    hn_hat = hn_hat + hk_xnk
                    return tf.add(joint, 1), n_joints, joints, hn_hat
                _, _, _, hn_hat = tf.while_loop(cond_j, body_j, [0, n_joints, joints, hn_hat])
                hn_hat = hn_hat / tf.cast(n_joints, dtype=tf.float32)
                # loop joints
                return tf.add(people, 1), n_people, tagmaps_gt_single, tagmaps_p_single
            tf.while_loop(cond_p, body_p, [0, n_people, tagmaps_gt_single, tagmaps_p_single])
            # loop peoples
            return tf.add(batch, 1), batch_size, tagmaps_gt, tagmaps_p          
        tf.while_loop(cond_b, body_b, [0, batch_size, tagmaps_gt, tagmaps_p])
        # loop batches
        return heatmaps_loss
    
    pass