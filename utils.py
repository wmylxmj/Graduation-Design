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
from PIL import Image, ImageDraw
import random
from configparser import ConfigParser

class KeypointsAnnotationsLoader(object):
    '''MS COCO'''
    
    def __init__(self, fp):
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
    
    def heatmaps(self, h, w, keypoints, num_joints=17, sigma=10):
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
        
    def sample(self, dataset, image_id, sigma=10):
        # visualize a sample
        image_name = "0" * (12-len(str(image_id))) + str(image_id) + ".jpg"
        image_path = os.path.join(dataset, image_name)
        keypoints = self.arrangements[str(image_id)]
        image = plt.imread(image_path)
        h = image.shape[0]
        w = image.shape[1]
        heatmaps = self.heatmaps(h, w, keypoints, sigma)
        image = np.uint32(image)
        for i in range(heatmaps.shape[2]):
            image[:, :, 0] = image[:, :, 0] + 64 * heatmaps[:, :, i]
            pass
        image = np.uint8(np.rint(np.minimum(image, 255)))
        plt.imshow(image)
        plt.show()
        pass
    
    pass


class SPEDataLoader(object):
    '''single pe data loader'''
    
    def __init__(self, images, annotations, image_size=(512, 512), scale=2):
        self.annotationsLoader = KeypointsAnnotationsLoader(fp=annotations)
        self.dataset = images
        self.image_size = image_size
        self.scale = scale
        pass
    
    def imread(self, fp):
        image = Image.open(fp)
        # asset a 3 ndim array
        image = np.array(image)
        if image.ndim == 2:
            image = np.repeat(np.expand_dims(image, 2), 3, axis=2)
            pass
        image = Image.fromarray(image)
        return image
    
    def pair(self, image_id):
        # image
        image_name = "0" * (12-len(str(image_id))) + str(image_id) + ".jpg"
        image_path = os.path.join(self.dataset, image_name)
        image = self.imread(image_path)
        original_w, original_h = image.size
        output_size = (self.image_size[0] // self.scale, self.image_size[1] // self.scale)
        # resize image
        image_resized = image.resize(self.image_size)
        image_resized = np.array(image_resized)
        # keypoints
        keypoints = self.annotationsLoader.arrangements[str(image_id)]
        num_people = len(keypoints)
        # resize keypoints
        keypoints_resized = []
        num_joints = 17
        for people in range(num_people):
            keypoints_resized.append([])
            for joint in range(num_joints):
                keypoint = keypoints[people][3*joint:3*joint+3]
                keypoints_resized[people].append(int(keypoint[0]*output_size[0]/original_w))
                keypoints_resized[people].append(int(keypoint[1]*output_size[1]/original_h))
                keypoints_resized[people].append(int(keypoint[2]))
                pass
            pass
        # heatmaps
        heatmaps = self.annotationsLoader.heatmaps(output_size[1], output_size[0], keypoints_resized, sigma=10//self.scale)
        return image_resized, heatmaps
    
    def sample(self):
        '''for predict'''
        image_ids = list(self.annotationsLoader.arrangements.keys())
        # random shuffle
        np.random.shuffle(image_ids)
        image_id = random.choice(image_ids)
        # image
        image_name = "0" * (12-len(str(image_id))) + str(image_id) + ".jpg"
        image_path = os.path.join(self.dataset, image_name)
        image = self.imread(image_path)
        # input tensor
        X = np.array(image)
        X = np.float32([X])
        # resize image
        image_resized = image.resize((image.size[0]//2, image.size[1]//2))
        image_resized = np.array(image_resized)
        return X, image_resized
    
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
            X, Y = [], []
            for image_id in batch:
                # pair
                image, heatmaps = self.pair(image_id)
                X.append(image)
                Y.append(heatmaps)
                pass
            # transfer to float32
            X = np.array(X, dtype=np.float32)
            Y = np.array(Y, dtype=np.float32)
            yield X, Y
        if self.n_batches > n_complete_batches:
            batch = image_ids[n_complete_batches*batch_size:]
            X, Y = [], []
            for image_id in batch:
                # pair
                image, heatmaps = self.pair(image_id)
                X.append(image)
                Y.append(heatmaps)
                pass
            # transfer to float32
            X = np.array(X, dtype=np.float32)
            Y = np.array(Y, dtype=np.float32)
            yield X, Y
        pass
    
    pass
        

class TCPDataLoader(object):
    
    def __init__(self):
        # dataset path
        cfg = ConfigParser()  
        cfg.read("config.cfg", encoding='utf-8')
        self.train_txt  = cfg.get("tcp_data", "train_txt")
        self.train_hot_data  = cfg.get("tcp_data", "train_hot_data")
        self.train_cold_data = cfg.get("tcp_data", "train_cold_data")
        self.val_txt  = cfg.get("tcp_data", "val_txt")
        self.val_hot_data  = cfg.get("tcp_data", "val_hot_data")
        self.val_cold_data = cfg.get("tcp_data", "val_cold_data")
        self.category_txt = cfg.get("tcp_data", "category_txt")
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], \
                         [12, 13], [6, 12], [7, 13], [6, 7], \
                         [6, 8], [7, 9], [8, 10], [9, 11], \
                         [2, 3], [1, 2], [1, 3], [2, 4], \
                         [3, 5], [4, 6], [5, 7]]
        self.n_frames = self.get_n_frames()
        pass
    
    def get_n_frames(self):
        # read train txt
        with open(self.train_txt, "r") as f:
            lines = f.readlines()
            pass
        train_txt_array = []
        for line in lines:
            i, f, c = line.split()    
            if c in ["0", "1", "2", "14", "15", "16"]:
                train_txt_array.append([int(i), int(f), int(c)])         
                pass
            pass
        train_txt_array = np.uint8(train_txt_array)
        # val txt
        with open(self.val_txt, "r") as f:
            lines = f.readlines()
            pass
        val_txt_array = []
        for line in lines:
            i, f, c = line.split()           
            if c in ["0", "1", "2", "14", "15", "16"]:
                val_txt_array.append([int(i), int(f), int(c)])   
                pass
            pass
        val_txt_array = np.uint8(val_txt_array)
        n_frames = max([np.squeeze(np.max(train_txt_array[:, 1])), \
                        np.squeeze(np.max(val_txt_array[:, 1]))])
        return n_frames
    
    def imread(self, fp):
        image = Image.open(fp)
        # asset a 3 ndim array
        image = np.array(image)
        if image.ndim == 2:
            image = np.repeat(np.expand_dims(image, 2), 3, axis=2)
            pass
        image = Image.fromarray(image)
        return image
    
    def prepare(self, model):
        # train
        with open(self.train_txt, "r") as f:
            lines = f.readlines()
            pass
        for line in lines:
            i, f, c = line.split()
            # hot
            if c in ["0", "1", "2"]:
                folder = os.path.join(self.train_hot_data, i)
                imgs = os.listdir(folder)
                imgs.sort()
                sequence = []
                data = {}
                for img in imgs:
                    img_path = os.path.join(folder, img)
                    image = self.imread(img_path)
                    # input tensor
                    X = np.array(image)
                    X = np.float32([X])
                    heatmaps = model.predict(X)[0]
                    joints = heatmaps.shape[-1]
                    vector = []
                    for joint in range(joints):
                        heatmap = heatmaps[:, :, joint]
                        point = np.unravel_index(heatmap.argmax(), heatmap.shape)
                        # x, y, p
                        vector.append(int(point[1]))
                        vector.append(int(point[0]))
                        vector.append(float(heatmaps[point[0]][point[1]][joint]))
                        pass
                    sequence.append(vector)
                    pass
                data["x"] = sequence
                data["y"] = int(c)
                with open("infos/train/{}.json".format(str(i)), "w") as f:
                    json.dump(data, f)
                    print("infos/train/{}.json writed.".format(str(i)))
                    pass
                pass
            # cold
            if c in ["14", "15", "16"]:
                folder = os.path.join(self.train_cold_data, i)
                imgs = os.listdir(folder)
                imgs.sort()
                sequence = []
                data = {}
                for img in imgs:
                    img_path = os.path.join(folder, img)
                    image = self.imread(img_path)
                    # input tensor
                    X = np.array(image)
                    X = np.float32([X])
                    heatmaps = model.predict(X)[0]
                    joints = heatmaps.shape[-1]
                    vector = []
                    for joint in range(joints):
                        heatmap = heatmaps[:, :, joint]
                        point = np.unravel_index(heatmap.argmax(), heatmap.shape)
                        vector.append(int(point[1]))
                        vector.append(int(point[0]))
                        vector.append(float(heatmaps[point[0]][point[1]][joint]))
                        pass
                    sequence.append(vector)
                    pass
                data["x"] = sequence
                data["y"] = int(c)
                with open("infos/train/{}.json".format(str(i)), "w") as f:
                    json.dump(data, f)
                    print("infos/train/{}.json writed.".format(str(i)))
                    pass
                pass
            pass
        # val
        with open(self.val_txt, "r") as f:
            lines = f.readlines()
            pass
        for line in lines:
            i, f, c = line.split()
            # hot
            if c in ["0", "1", "2"]:
                folder = os.path.join(self.val_hot_data, i)
                imgs = os.listdir(folder)
                imgs.sort()
                sequence = []
                data = {}
                for img in imgs:
                    img_path = os.path.join(folder, img)
                    image = self.imread(img_path)
                    # input tensor
                    X = np.array(image)
                    X = np.float32([X])
                    heatmaps = model.predict(X)[0]
                    joints = heatmaps.shape[-1]
                    vector = []
                    for joint in range(joints):
                        heatmap = heatmaps[:, :, joint]
                        point = np.unravel_index(heatmap.argmax(), heatmap.shape)
                        vector.append(int(point[1]))
                        vector.append(int(point[0]))
                        vector.append(float(heatmaps[point[0]][point[1]][joint]))
                        pass
                    sequence.append(vector)
                    pass
                data["x"] = sequence
                data["y"] = int(c)
                with open("infos/val/{}.json".format(str(i)), "w") as f:
                    json.dump(data, f)
                    print("infos/val/{}.json writed.".format(str(i)))
                    pass
                pass
            # cold
            if c in ["14", "15", "16"]:
                folder = os.path.join(self.val_cold_data, i)
                imgs = os.listdir(folder)
                imgs.sort()
                sequence = []
                data = {}
                for img in imgs:
                    img_path = os.path.join(folder, img)
                    image = self.imread(img_path)
                    # input tensor
                    X = np.array(image)
                    X = np.float32([X])
                    heatmaps = model.predict(X)[0]
                    joints = heatmaps.shape[-1]
                    vector = []
                    for joint in range(joints):
                        heatmap = heatmaps[:, :, joint]
                        point = np.unravel_index(heatmap.argmax(), heatmap.shape)
                        vector.append(int(point[1]))
                        vector.append(int(point[0]))
                        vector.append(float(heatmaps[point[0]][point[1]][joint]))
                        pass
                    sequence.append(vector)
                    pass
                data["x"] = sequence
                data["y"] = int(c)
                with open("infos/val/{}.json".format(str(i)), "w") as f:
                    json.dump(data, f)
                    print("infos/val/{}.json writed.".format(str(i)))
                    pass
                pass
            pass
        pass
    
    def visualize(self, json_path="infos/train/0.json", images_folder="F:/热舒适姿态data/train/0", frame=0):
        with open(json_path, "r") as f: 
            data = json.load(f)
            pass
        images = os.listdir(images_folder)
        images.sort()
        points = data['x'][frame]
        img_path = os.path.join(images_folder, images[frame])
        image = self.imread(img_path)
        image = image.resize((image.size[0]//2, image.size[1]//2))
        draw = ImageDraw.Draw(image)
        for line in self.skeleton:
            c1 = line[0] - 1
            c2 = line[1] - 1
            p1 = tuple(points[3*c1:3*c1+2])
            p2 = tuple(points[3*c2:3*c2+2])
            draw.line([p1, p2], fill='red', width=1)
            pass
        image = np.array(image)
        plt.imshow(image)
        plt.show()
        pass
    
    def pair(self, json_path, n_classes=6, test_mode=False):
        with open(json_path, "r") as f: 
            data = json.load(f)
            pass
        x = data['x']
        y = data['y']
        if y == 14:
            y = 3
            pass
        elif y == 15:
            y = 4
            pass
        elif y == 16:
            y = 5
            pass
        y = np.eye(n_classes)[np.squeeze(y)]
        w_min = np.min((np.float32(x)[:, 0::3]))
        w_max = np.max((np.float32(x)[:, 0::3]))
        h_min = np.min((np.float32(x)[:, 1::3]))
        h_max = np.max((np.float32(x)[:, 1::3]))
        x = np.float32(x)
        # in train mode
        if test_mode==False:
            rotate_center = [153, 100]
            rotate_center[0] = rotate_center[0] + random.sample(range(-50, 50), 1)[0]
            rotate_center[1] = rotate_center[1] + random.sample(range(-25, 25), 1)[0]
            angle = random.sample(range(-30, 30), 1)[0] / 180.0 * np.pi
            x[:, 0::3] -= rotate_center[0]
            x[:, 1::3] -= rotate_center[1]
            # rotate
            distance = (x[:, 0::3]**2 + x[:, 1::3]**2) ** 0.5
            atan = np.arctan(x[:, 1::3]/x[:, 0::3])
            # y<0 x<0 or y>0 x<0
            atan += np.float32(x[:, 1::3]<0) * np.float32(x[:, 0::3]<0) * np.pi + \
            np.float32(x[:, 1::3]>0) * np.float32(x[:, 0::3]<0) * np.pi
            angle += atan
            x[:, 0::3] = distance * np.cos(angle)
            x[:, 1::3] = distance * np.sin(angle)
            # restore
            x[:, 0::3] += rotate_center[0]
            x[:, 1::3] += rotate_center[1]
            pass
        # w
        x[:, 0::3] -= w_min
        x[:, 0::3] /= (w_max - w_min)
        # h
        x[:, 1::3] -= h_min
        x[:, 1::3] /= (h_max - h_min)
        x = x.tolist()
        n_features = len(x[0])
        # padding
        if len(x) < self.n_frames:
            for i in range(self.n_frames-len(x)):
                x.append([0 for j in range(n_features)])
                pass
            pass
        x = np.float32(x[-1*self.n_frames:])
        y = np.float32(y)
        return x, y
    
    def batches(self, batch_size=4, complete_batch_only=False, folder="infos/train", test_mode=False):
        files = os.listdir(folder)
        jsons = list(filter(lambda f: f[-5:]==".json", files))
        # random shuffle
        np.random.shuffle(jsons)
        # compute n batches
        n_complete_batches = int(len(jsons) / batch_size)
        self.n_batches = int(len(jsons) / batch_size)
        have_res_batch = (len(jsons) / batch_size) > n_complete_batches
        if have_res_batch and complete_batch_only==False:
            self.n_batches += 1
            pass
        # yield
        for i in range(n_complete_batches):
            batch = jsons[i*batch_size:(i+1)*batch_size]
            X, Y = [], []
            for file in batch:
                path = os.path.join(folder, file)
                # pair
                x, y = self.pair(path, test_mode=test_mode)
                X.append(x)
                Y.append(y)
                pass
            # transfer to float32
            X = np.array(X, dtype=np.float32)
            Y = np.array(Y, dtype=np.float32)
            yield X, Y
        if self.n_batches > n_complete_batches:
            batch = jsons[n_complete_batches*batch_size:]
            X, Y = [], []
            for file in batch:
                path = os.path.join(folder, file)
                # pair
                x, y = self.pair(path, test_mode=test_mode)
                X.append(x)
                Y.append(y)
                pass
            # transfer to float32
            X = np.array(X, dtype=np.float32)
            Y = np.array(Y, dtype=np.float32)
            yield X, Y
        pass
    
    pass