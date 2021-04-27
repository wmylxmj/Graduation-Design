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
                # 0 means unlabeled
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


class DataLoader(object):
    
    def __init__(self, images="D:/train2017", annotations="D:/annotations/person_keypoints_train2017.json", image_size=(640, 480), scale=2):
        self.annotationsLoader = AnnotationsLoader(fp=annotations)
        self.dataset = images
        self.image_size = image_size
        self.scale = scale
        pass
    
    def imread(self, fp):
        return Image.open(fp)
    
    def pair(self, image_id):
        # image
        image_name = "0" * (12-len(str(image_id))) + str(image_id) + ".jpg"
        image_path = os.path.join(self.dataset, image_name)
        image = self.imread(image_path)
        (w, h) = image.size
        # keypoints
        keypoints = self.annotationsLoader.arrangements[str(image_id)]
        num_people = len(keypoints)
        # heatmaps
        heatmaps = self.annotationsLoader.heatmap(h, w, keypoints)
        num_joints = heatmaps.shape[2]
        # resize image
        image_resized = image.resize(self.image_size)
        image_resized = np.array(image_resized)
        # resize keypoints
        output_size = (self.image_size[0] // self.scale, self.image_size[1] // self.scale)
        keypoints_resized = []
        keypoints = self.annotationsLoader.arrangements[str(image_id)]
        num_people = len(keypoints)
        for people in range(num_people):
            keypoints_resized.append([])
            for joint in range(num_joints):
                keypoint = keypoints[people][3*joint:3*joint+3]
                keypoints_resized[people].append(int(keypoint[0]*output_size[0]/w))
                keypoints_resized[people].append(int(keypoint[1]*output_size[1]/h))
                keypoints_resized[people].append(int(keypoint[2]))
                pass
            pass
        # resize heatmaps
        heatmaps_resized = np.zeros((output_size[1], output_size[0], heatmaps.shape[2]), dtype=np.float32)
        for i in range(num_joints):
            heatmap = Image.fromarray(heatmaps[:, :, i])
            heatmap = heatmap.resize(output_size)
            heatmaps_resized[:, :, i] = np.array(heatmap)
            pass     
        return image_resized, heatmaps_resized, keypoints_resized
    
    def batches(self, batch_size=4, complete_batch_only=False):
        image_ids = list(self.annotationsLoader.arrangements.keys())
        # random shuffle
        np.random.shuffle(image_ids)
        # compute n batches
        n_complete_batches = int(len(image_ids)/batch_size)
        self.n_batches = int(len(image_ids) / batch_size)
        have_res_batch = (len(image_ids)/batch_size) > n_complete_batches
        if have_res_batch and complete_batch_only==False:
            self.n_batches += 1
            pass
        # yield
        for i in range(n_complete_batches):
            batch = image_ids[i*batch_size:(i+1)*batch_size]
            I, H, K = [], [], []
            for image_id in batch:
                # pair
                image, heatmaps, keypoints = self.pair(image_id)
                I.append(image)
                H.append(heatmaps)
                K.append(keypoints)
                pass
            # transfer to float32
            I = np.float32(I)
            H = np.float32(H)
            yield I, (H, K)
        if self.n_batches > n_complete_batches:
            batch = image_ids[n_complete_batches*batch_size:]
            I, H, K = [], [], []
            for image_id in batch:
                # pair
                image, heatmaps, keypoints = self.pair(image_id)
                I.append(image)
                H.append(heatmaps)
                K.append(keypoints)
                pass
            # transfer to float32
            I = np.float32(I)
            H = np.float32(H)
            yield I, (H, K)
        pass
    
    pass
        


