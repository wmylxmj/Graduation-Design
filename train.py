# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:58:17 2021

@author: wmy
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from configparser import ConfigParser
from PIL import Image
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import load_model
from keras.optimizers import Adam
import random
from keras.layers import Lambda
from optimizer import AdamWithWeightNorm
from utils import SPEDataLoader, TCPDataLoader
from model import SPENet, TCPNet

# dataset path
cfg = ConfigParser()  
cfg.read("config.cfg", encoding='utf-8')
annotations = cfg.get("spe_data", "train_annotations")
images = cfg.get("spe_data", "train_images")

class SPENetTrain(object):
    
    def __init__(self, layers=16, joints=17, lr=1e-4, pretrained_weights=None, name=None):
        self.model = SPENet(layers=layers, joints=joints)
        self.model.compile(optimizer=AdamWithWeightNorm(lr=lr), loss=self.loss)
        print("[OK] model created.")
        # load pretrained weights
        if pretrained_weights != None:
            self.model.load_weights(pretrained_weights)
            print("[OK] weights loaded.")
            pass
        self.pretrained_weights = pretrained_weights
        self.default_weights_save_path = 'weights/SPENet-{}-{}.h5'.format(layers, joints)
        # data loader
        self.data_loader = SPEDataLoader(images, annotations)
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
                if (batch_i+1) % 100 == 0:
                    self.model.save_weights(weights_save_path)
                    print("[OK] weights saved.")
                pass
            pass
        pass
    
    def sample(self, save_folder='samples', epoch=1, batch=1):
        X, image_resized = self.data_loader.sample()
        heatmaps = self.model.predict(X)[0]
        print("sample heatmaps max: {}".format(np.max(heatmaps)))
        image = np.float32(np.rint(image_resized))
        heatmap = np.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=np.float32)
        for i in range(heatmaps.shape[2]):
            image[:, :, 0] = image[:, :, 0] + 255 * heatmaps[:, :, i]
            heatmap[:, :, 0] = heatmap[:, :, 0] + 255 * heatmaps[:, :, i]
            heatmap[:, :, 1] = heatmap[:, :, 1] + 255 * heatmaps[:, :, i]
            heatmap[:, :, 2] = heatmap[:, :, 2] + 255 * heatmaps[:, :, i]
            pass
        image = np.uint8(np.rint(np.maximum(np.minimum(image, 255), 0)))
        image = Image.fromarray(image)
        heatmap = np.uint8(np.rint(np.maximum(np.minimum(heatmap, 255), 0)))
        heatmap = Image.fromarray(heatmap)
        oringal_image = np.uint8(np.rint(image_resized))
        oringal_image = Image.fromarray(oringal_image)
        oringal_image.save(save_folder + "/" + "epoch_" + str(epoch) + "_batch_" + str(batch) + "_oringal.jpg")
        heatmap.save(save_folder + "/" + "epoch_" + str(epoch) + "_batch_" + str(batch) + "_heatmap.jpg")
        image.save(save_folder + "/" + "epoch_" + str(epoch) + "_batch_" + str(batch) + "_image.jpg")
        pass
    
    pass


class TCPNetTrain(object):
    
    def __init__(self, features=51, ndim=6, lr=2e-5, pretrained_weights=None, name=None):
        # data loader
        self.data_loader = TCPDataLoader()
        step = self.data_loader.n_frames
        # model
        self.model = TCPNet(step=step, features=features, ndim=6)
        self.model.compile(optimizer=AdamWithWeightNorm(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
        print("[OK] model created.")
        # load pretrained weights
        if pretrained_weights != None:
            self.model.load_weights(pretrained_weights)
            print("[OK] weights loaded.")
            pass
        self.pretrained_weights = pretrained_weights
        self.default_weights_save_path = 'weights/TCPNet-{}-{}-{}.h5'.format(step, features, ndim)
        self.name = name
        pass
    
    def train(self, epoches=10000, batch_size=8, weights_save_path=None):
        # save path
        if weights_save_path == None:
            weights_save_path = self.default_weights_save_path
            pass
        # train
        for epoch in range(epoches):
            for batch_i, (X, Y) in enumerate(self.data_loader.batches(batch_size=batch_size)):
                temp_loss, temp_accuracy = self.model.train_on_batch(X, Y)
                print("[epoch: {}/{}][batch: {}/{}][loss: {}][accuracy: {}]".format(epoch+1, epoches, \
                      batch_i+1, self.data_loader.n_batches, temp_loss, temp_accuracy))
                pass
            self.model.save_weights(weights_save_path)
            print("[OK] weights saved.")
            if ((epoch+1) % 10) == 0:                
                self.evaluate(batch_size=64)
                pass
            pass
        pass
    
    def evaluate(self, batch_size=32, folder="infos/val"):
        loss = 0
        acc = 0
        num = 0
        for batch_i, (X, Y) in enumerate(self.data_loader.batches(batch_size=batch_size, folder=folder, test_mode=True)):
            n = X.shape[0]
            temp_loss, temp_acc = self.model.evaluate(X, Y)
            print("[batch: {}/{}][loss: {}][accuracy: {}]".format(batch_i+1, self.data_loader.n_batches, \
                  temp_loss, temp_acc))
            num += n
            loss += temp_loss * n
            acc += temp_acc * n
            pass
        loss = loss / num
        acc = acc / num
        print("[total loss: {}][total accuracy: {}]".format(loss, acc))
        return loss, acc
            
    pass

t = TCPNetTrain(pretrained_weights="weights/TCPNet-136-51-6.h5")
t.train(batch_size=64)