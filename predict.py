# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:47:17 2021

@author: wmy
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import load_model
from keras.optimizers import Adam
import random
from keras.layers import Lambda
from optimizer import AdamWithWeightNorm
from utils import DataLoader
from model import SPENet

model = SPENet(layers=8)
model.load_weights("weights/SPENet-8-17-202105132055.h5")

class SPENetPredict(object):
    
    def __init__(self, model):
        self.model = model
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], \
                         [12, 13], [6, 12], [7, 13], [6, 7], \
                         [6, 8], [7, 9], [8, 10], [9, 11], \
                         [2, 3], [1, 2], [1, 3], [2, 4], \
                         [3, 5], [4, 6], [5, 7]]
        pass
    
    def get_skeleton_points(self, heatmaps):
        skeleton = []
        for line in  self.skeleton:
            channel1 = line[0] - 1
            channel2 = line[1] - 1
            heatmap1 = heatmaps[:, :, channel1]
            heatmap2 = heatmaps[:, :, channel2]
            point1 = np.unravel_index(heatmap1.argmax(), heatmap1.shape)
            point2 = np.unravel_index(heatmap2.argmax(), heatmap2.shape)
            skeleton.append([point1, point2])
            pass
        return skeleton
            
    def predict_skeleton(self, img_path, save_folder="outputs", save_name="test", scale=2):
        image = Image.open(img_path)
        # asset a 3 ndim array
        image = np.array(image)
        if image.ndim == 2:
            image = np.repeat(np.expand_dims(image, 2), 3, axis=2)
            pass
        image = Image.fromarray(image)
        # input tensor
        X = np.array(image)
        X = np.float32([X])
        heatmaps = self.model.predict(X)[0]
        skeleton = self.get_skeleton_points(heatmaps)
        draw = ImageDraw.Draw(image)
        for points in skeleton:
            point1 = (points[0][1]*image.size[0]/heatmaps.shape[1], points[0][0]*image.size[1]/heatmaps.shape[0])
            point2 = (points[1][1]*image.size[0]/heatmaps.shape[1], points[1][0]*image.size[1]/heatmaps.shape[0])
            draw.line([point1, point2], fill ="red", width = 1)
            pass
        image.save(save_folder + "/" + save_name + "_skeleton.jpg")
        pass

    def predict_heatmap(self, img_path, save_folder="outputs", save_name="test"):
        image = Image.open(img_path)
        # asset a 3 ndim array
        image = np.array(image)
        if image.ndim == 2:
            image = np.repeat(np.expand_dims(image, 2), 3, axis=2)
            pass
        image = Image.fromarray(image)
        # input tensor
        X = np.array(image)
        X = np.float32([X])
        # resize image
        image_resized = image.resize((image.size[0]//2, image.size[1]//2))
        image_resized = np.array(image_resized)
        heatmaps = self.model.predict(X)[0]
        print("predict heatmaps max: {}".format(np.max(heatmaps)))
        image = np.float32(np.rint(image_resized))
        heatmap = np.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=np.float32)
        for i in range(heatmaps.shape[2]):
            image[:, :, 0] = image[:, :, 0] + 64 * heatmaps[:, :, i]
            heatmap[:, :, 0] = heatmap[:, :, 0] + 64 * heatmaps[:, :, i]
            heatmap[:, :, 1] = heatmap[:, :, 1] + 64 * heatmaps[:, :, i]
            heatmap[:, :, 2] = heatmap[:, :, 2] + 64 * heatmaps[:, :, i]
            pass
        image = np.uint8(np.rint(np.maximum(np.minimum(image, 255), 0)))
        image = Image.fromarray(image)
        heatmap = np.uint8(np.rint(np.maximum(np.minimum(heatmap, 255), 0)))
        heatmap = Image.fromarray(heatmap)
        oringal_image = np.uint8(np.rint(image_resized))
        oringal_image = Image.fromarray(oringal_image)
        oringal_image.save(save_folder + "/" + save_name + "_oringal.jpg")
        heatmap.save(save_folder + "/" + save_name + "_heatmap.jpg")
        image.save(save_folder + "/" + save_name + "_image.jpg")
        pass


p = SPENetPredict(model)
for i in range(47):
    fp = os.path.join("D:/热舒适姿态data/train/0/", "0"*(5-len(str(i+1)))+str(i+1)+".jpg")
    p.predict_skeleton("D:/热舒适姿态data/train/0/00020.jpg", save_name="0"*(5-len(str(i+1)))+str(i+1))