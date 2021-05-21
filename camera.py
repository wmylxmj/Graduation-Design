# -*- coding: utf-8 -*-
"""
Created on Fri May 21 21:08:15 2021

@author: wmy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from model import SPENet, TCPNet
import time

def get_skeleton_points(heatmaps, visual_threshold=0.5):
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], \
                [12, 13], [6, 12], [7, 13], [6, 7], \
                [6, 8], [7, 9], [8, 10], [9, 11], \
                [2, 3], [1, 2], [1, 3], [2, 4], \
                [3, 5], [4, 6], [5, 7]]
    visual = []
    for line in skeleton:
        channel1 = line[0] - 1
        channel2 = line[1] - 1
        heatmap1 = heatmaps[:, :, channel1]
        heatmap2 = heatmaps[:, :, channel2]
        point1 = np.unravel_index(heatmap1.argmax(), heatmap1.shape)
        point2 = np.unravel_index(heatmap2.argmax(), heatmap2.shape)
        if heatmap1[point1[0]][point1[1]] < visual_threshold or heatmap2[point2[0]][point2[1]] < visual_threshold:
            visual.append(None)
            pass
        else:
            p1 = (point1[1]*2, point1[0]*2)
            p2 = (point2[1]*2, point2[0]*2)
            visual.append([p1, p2])
            pass
        pass
    return visual

def get_pose_vector(heatmaps):
    vector = []
    for joint in range(heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, joint]
        point = np.unravel_index(heatmap.argmax(), heatmap.shape)
        vector.append(point[1])
        vector.append(point[0])
        vector.append(heatmap[point[0]][point[1]])
        pass
    return vector
        
def hsv2rgb(h, s, v):
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
    r = int(np.uint8(np.rint(np.maximum(np.minimum(r, 255), 0))))
    g = int(np.uint8(np.rint(np.maximum(np.minimum(g, 255), 0))))
    b = int(np.uint8(np.rint(np.maximum(np.minimum(b, 255), 0))))
    return r, g, b

def get_colors(num, mode="rgb"):
    hs = [360 / num * i + 0.5 * 360 / num for i in range(num)]
    colors = []
    for h in hs:
        r, g, b = hsv2rgb(h, s=1, v=1)
        if mode == "rgb" or mode == "RGB":
            colors.append((r, g, b))
            pass
        elif mode == "bgr" or mode == "BGR":
            colors.append((b, g, r))
            pass
        else:
            raise Exception("Unkown mode.") 
            pass
        pass
    return colors

category = {}
category["0"] = "Wiping sweat"
category["1"] = "Fanning with hands"
category["2"] = "Shaking T-shirt"
category["3"] = "Rubbing hands"
category["4"] = "Collar tuggingt"
category["5"] = "Contracted posture"

print("CREATING SPENET...")
spe = SPENet(layers=8, joints=17)
print("SPENET LOADING WEIGHTS...")
spe.load_weights("weights/SPENet-8-17.h5")    
print("CREATING TCPNET...")
tcp = TCPNet(step=136, features=51, ndim=6)
print("TCPNET LOADING WEIGHTS...")
tcp.load_weights("weights/TCPNet-136-51-6.h5")    

camera = cv2.VideoCapture(0)
# skeleton has 19 lines
colors = get_colors(19, mode="BGR")

video_length = 15
sequence = []
for i in range(video_length):
    sequence.append([0.0 for j in range(51)])
    pass

heatmap_threshold = 0.4

while True:
    see_whole_body = True
    ret, frame = camera.read()
    # input  BGR TO RGB
    X = np.array(frame[...,::-1])
    X = np.float32([X])
    # predict heatmap
    heatmaps = spe.predict(X)[0]
    skeleton = get_skeleton_points(heatmaps, heatmap_threshold)
    pose_vector = get_pose_vector(heatmaps)
    sequence.insert(0, pose_vector)
    sequence.remove(sequence[video_length])
    # process
    s = np.float32(sequence)
    w_min = np.min(s[:, 0::3])
    w_max = np.max(s[:, 0::3])
    h_min = np.min(s[:, 1::3])
    h_max = np.max(s[:, 1::3])
    # w
    s[:, 0::3] -= w_min
    s[:, 0::3] /= (w_max - w_min)
    # h
    s[:, 1::3] -= h_min
    s[:, 1::3] /= (h_max - h_min)
    s = s.tolist()
    # padding step = 136
    for i in range(136-len(s)):
        s.append([0 for j in range(51)])
        pass
    s = np.float32([s[-1*136:]])
    # draw lines
    for i, line in enumerate(skeleton):
        if line == None:
            see_whole_body = False
            continue
        cv2.line(frame, line[0], line[1], colors[i], thickness=2)
        pass
    y = tcp.predict(s)[0]
    index = np.argmax(y)
    cv2.putText(frame, "{}: {}".format(category[str(index)], y[index]), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("camera", frame)
    cv2.imwrite("video/{}.jpg".format(time.time()), frame)
    # exit camera
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
    pass

camera.release()
cv2.destroyAllWindows()