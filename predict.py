# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:47:17 2021

@author: wmy
"""

import os
import json
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import load_model
from keras.optimizers import Adam
import random
from tqdm import tqdm
from keras.layers import Lambda
from optimizer import AdamWithWeightNorm
from utils import DataLoader
from model import SPENet

model = SPENet(layers=8)
model.load_weights("weights/SPENet-8-17.h5")

class SPENetPredict(object):
    
    def __init__(self, model):
        self.model = model
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], \
                         [12, 13], [6, 12], [7, 13], [6, 7], \
                         [6, 8], [7, 9], [8, 10], [9, 11], \
                         [2, 3], [1, 2], [1, 3], [2, 4], \
                         [3, 5], [4, 6], [5, 7]]
        self.colors = self.get_colors(num=len(self.skeleton))
        pass
    
    def hsv2rgb(self, h, s, v):
        # h 0-360 s 0-1 v 0-1
        h = h % 360
        c = v * s
        # x = c * (1 - |(h/60) mod 2 - 1|)
        x = c * (1 - abs((h / 60.0) % 2 - 1))
        m = v - c
        if (h >= 0 and h < 60) or h==360:
            r, g, b = c, x, 0
            pass
        elif h >= 60 and h < 120:
            r, g, b = x, c, 0
            pass
        elif h >= 120 and h < 180:
            r, g, b = 0, c, x
            pass
        elif h >= 180 and h < 240:
            r, g, b = 0, x, c
            pass
        elif h >= 240 and h < 300:
            r, g, b = x, 0, c
            pass
        elif h >= 300 and h < 360:
            r, g, b = c, 0, x
            pass
        r, g, b = (r + m) *255, (g + m) *255, (b + m) *255
        # to uint8
        r = np.uint8(np.rint(np.maximum(np.minimum(r, 255), 0)))
        g = np.uint8(np.rint(np.maximum(np.minimum(g, 255), 0)))
        b = np.uint8(np.rint(np.maximum(np.minimum(b, 255), 0)))
        return r, g, b
    
    def get_colors(self, num):
        hs = [360 / num * i + 0.5 * 360 / num for i in range(num)]
        colors = []
        for h in hs:
            r, g, b = self.hsv2rgb(h, s=1, v=1)
            r = "0" * (2 - len(hex(r)[2:])) + hex(r)[2:]
            g = "0" * (2 - len(hex(g)[2:])) + hex(g)[2:]
            b = "0" * (2 - len(hex(b)[2:])) + hex(b)[2:]
            color = "#{}{}{}".format(r, g, b)
            colors.append(color)
            pass
        return colors
    
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
        for i, points in enumerate(skeleton):
            point1 = (points[0][1]*image.size[0]/heatmaps.shape[1], points[0][0]*image.size[1]/heatmaps.shape[0])
            point2 = (points[1][1]*image.size[0]/heatmaps.shape[1], points[1][0]*image.size[1]/heatmaps.shape[0])
            draw.line([point1, point2], fill =self.colors[i], width=2)
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
for i in tqdm(range(47)):
    fp = os.path.join("F:/热舒适姿态data/train/0/", "0"*(5-len(str(i+1)))+str(i+1)+".jpg")
    p.predict_skeleton(fp, save_name="0_"+"0"*(5-len(str(i+1)))+str(i+1))
    pass
for i in tqdm(range(47)):
    fp = os.path.join("F:/热舒适姿态data/train/583/", "0"*(5-len(str(i+1)))+str(i+1)+".jpg")
    p.predict_skeleton(fp, save_name="1_"+"0"*(5-len(str(i+1)))+str(i+1))
    pass
for i in tqdm(range(41)):
    fp = os.path.join("F:/热舒适姿态data/train/1166/", "0"*(5-len(str(i+1)))+str(i+1)+".jpg")
    p.predict_skeleton(fp, save_name="2_"+"0"*(5-len(str(i+1)))+str(i+1))
    pass
for i in tqdm(range(38)):
    fp = os.path.join("F:/热舒适姿态data/train2/5436/", "0"*(5-len(str(i+1)))+str(i+1)+".jpg")
    p.predict_skeleton(fp, save_name="3_"+"0"*(5-len(str(i+1)))+str(i+1))
    pass
for i in tqdm(range(60)):
    fp = os.path.join("F:/热舒适姿态data/train2/5974/", "0"*(5-len(str(i+1)))+str(i+1)+".jpg")
    p.predict_skeleton(fp, save_name="4_"+"0"*(5-len(str(i+1)))+str(i+1))
    pass
for i in tqdm(range(49)):
    fp = os.path.join("F:/热舒适姿态data/train2/6513/", "0"*(5-len(str(i+1)))+str(i+1)+".jpg")
    p.predict_skeleton(fp, save_name="5_"+"0"*(5-len(str(i+1)))+str(i+1))
    pass

import cv2
videowrite = cv2.VideoWriter('output.mp4', -1, 10, (306, 201))
for fn in os.listdir("outputs"):
    img = cv2.imread(os.path.join("outputs", fn))
    videowrite.write(img)
    pass
videowrite.release()
cv2.destroyAllWindows()