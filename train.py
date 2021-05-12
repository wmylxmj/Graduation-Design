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
from model import SAENet

class SAEModel(object):
    
    def __init__(self, layers=16, joints=17, pretrained_weights=None, name=None):
        self.model = SAENet(layers=layers, joints=joints)
        self.model.compile(optimizer=AdamWithWeightNorm(lr=0.001), loss=self.loss)
        print("[OK] model created.")
        # load pretrained weights
        if pretrained_weights != None:
            self.model.load_weights(pretrained_weights)
            print("[OK] weights loaded.")
            pass
        self.pretrained_weights = pretrained_weights
        self.default_weights_save_path = 'weights/SAENet-{}-{}.h5'.format(layers, joints)
        # data loader
        self.data_loader = DataLoader()
        self.name = name
        pass
    
    def loss(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    
    def train(self, epoches=10000, batch_size=8, weights_save_path=None):
        # save path
        if weights_save_path == None:
            weights_save_path = self.default_weights_save_path
            pass
        # train
        for epoch in range(epoches):
            for batch_i, (X, Y) in enumerate(self.data_loader.batches(batch_size=batch_size)):
                temp_loss = self.model.train_on_batch(X, Y)
                print("[epoch: {}/{}][batch: {}/{}][loss: {}]".format(epoch+1, epoches, \
                      batch_i+1, self.data_loader.n_batches, temp_loss))
                if (batch_i+1) % 25 == 0:
                    self.sample(epoch=epoch+1, batch=batch_i+1)                 
                    pass
                if (batch_i+1) % 50 == 0:
                    self.model.save_weights(weights_save_path)
                    print("[OK] weights saved.")
                pass
            pass
        pass
    
    def sample(self, save_folder='samples', epoch=1, batch=1):
        X, image_resized = self.data_loader.sample()
        heatmaps = self.model.predict(X)[0]
        image = np.uint8(np.rint(image_resized))
        for i in range(heatmaps.shape[2]):
            image[:, :, 0] = image[:, :, 0] + 64 * heatmaps[:, :, i]
            pass
        image = np.uint8(np.rint(np.minimum(image, 255)))
        image = Image.fromarray(image)
        image.save(save_folder + "/" + "epoch_" + str(epoch) + "_batch_" + str(batch) + ".jpg")
        pass
    
    pass


sae = SAEModel(layers=8, pretrained_weights=None)
sae.train(batch_size=2)