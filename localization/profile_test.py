import pyrealsense2 as rs
import cv2
from detect import run, non_max_suppression
import numpy as np
import time
import torch
import io
import os

# TODO: 
# - Add platform/goal state detection
# - Add localization fusion logic
# - Find out why x1/y1 values are out of frame
# - Find out why max detections for NMS are always being reached
# - Optimize NMS code

focal_length = ((448.0-172.0) * 24.0) / 11.0
# ['Blue Goal', 'Blue Platform', 'Blue Robot', 'Neutral Goal', 'Red Goal', 'Red Platform', 'Red Robot', 'Ring']
width_dict = {"7":3.5, "3":12.5, "0":12.5, "4":12.5, "2":5.5, "5":5.5, "1":53.0, "6":53.0}

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

model = torch.jit.load('static/best_torchscript.pt', map_location=torch.device('cuda:0'))

def obj_distance(obj, depth_frame):
    # file parsing
    x1, y1, x2, y2, conf, cls = obj

    # calculating distance using trigonometric properties
    trig_distance = (width_dict[str(int(cls))] * focal_length)/x2-x1
    
    # calculating center of object
    x = (x1 + x2)/2 
    y = (y1 + y2)/2
    
    # extract average distance from depth map and convert to inches
    depth_distance_meters = (depth_frame.get_distance(x, y) +\
                             depth_frame.get_distance(x+2, y) +\
                             depth_frame.get_distance(x, y+2) +\
                             depth_frame.get_distance(x-2, y) +\
                             depth_frame.get_distance(x, y-2))/5.0
    depth_distance = 39.3701 * depth_distance_meters
    
    # weighting and combining localization methods
    distance = (trig_distance * .2) + (depth_distance_meters * .8) 
    
    # in the event that depthmap can't detect distance, only use trig distance
    if (depth_distance_meters == 0):
        distance = trig_distance
    
    return distance

t = 0
i = 0
while i < 103:
    if i == 3:
        t = time.time()
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    resized = cv2.resize(color_image, (640, 640), interpolation = cv2.INTER_AREA)
    model_input = torch.reshape(torch.tensor(resized).float(), (1, 3, 640, 640)).cuda()
    results = model(model_input)[0]
    nms_results = non_max_suppression(results, conf_thres=0.7)[0]


    # calculating distance for all game objects in frame
#     for obj in nms_results:
#         print(f'{obj.split()[0]}: {obj_distance(obj, depth_frame)}')
    i+=1
print(100/(time.time()-t),"fps")
