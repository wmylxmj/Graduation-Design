# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:29:08 2021

@author: wmy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from PIL import Image

class AnnotationsLoader(object):
    '''MS COCO'''
    
    def __init__(self, fp="D:/annotations/person_keypoints_train2017.json"):
        # load annotations
        self.fp = fp
        with open(fp, 'r') as file:
            file = file.read()
            self.json = json.loads(file)
            pass
        self.annotations = self.json['annotations']
        # process
        self.arrange()
        pass
    
    def arrange(self):
        # reorder annotations
        self.arrangements = {}
        for annotation in tqdm(self.annotations):
            image_id = annotation["image_id"]
            if str(image_id) not in self.arrangements.keys():
                self.arrangements[str(image_id)] = [annotation['keypoints']]
                pass
            else:
                self.arrangements[str(image_id)].append(annotation['keypoints'])
                pass
            pass
        pass
    
    def heatmap(self, h, w, keypoints, sigma=10):
        # ms coco have 17 joints
        num_joints = 17
        num_people = len(keypoints)
        heatmaps = np.zeros((h, w, num_joints), dtype=np.float32)
        yl = np.linspace(1, h, h)
        xl = np.linspace(1, w, w)
        [X, Y] = np.meshgrid(xl, yl)
        for joint in range(num_joints):
            G = np.zeros((h, w), dtype=np.float32)
            for people in range(num_people):
                keypoint = keypoints[people][3*joint:3*joint+3]
                # 0 means unvisible
                if keypoint[2] == 0:
                    continue
                cx = keypoint[0]
                cy = keypoint[1]
                g = np.exp(- ((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
                G += g
                pass
            # avoid 0 / 0
            if np.sum(G) != 0:
                G = (G - np.min(G)) / (np.max(G) - np.min(G))
                pass
            heatmaps[:, :, joint] = G
            pass
        return  heatmaps
    
    def sample(self, dataset="D:/train2017", image_id="386424", sigma=10):
        # visualize a sample
        image_name = "0" * (12-len(str(image_id))) + str(image_id) + ".jpg"
        image_path = os.path.join(dataset, image_name)
        keypoints = self.arrangements[str(image_id)]
        image = plt.imread(image_path)
        h = image.shape[0]
        w = image.shape[1]
        heatmaps = self.heatmap(h, w, keypoints, sigma)
        image = np.uint32(image)
        for i in range(heatmaps.shape[2]):
            image[:, :, 0] = image[:, :, 0] + 64 * heatmaps[:, :, i]
            pass
        image = np.uint8(np.rint(np.minimum(image, 255)))
        plt.imshow(image)
        plt.show()
        pass
    
    pass


a = AnnotationsLoader()
a.arrange()

