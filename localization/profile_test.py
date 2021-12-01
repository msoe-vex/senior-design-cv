import pyrealsense2 as rs
import cv2
from detect import run
import numpy as np
import time
import torch
import io
import os

focal_length = ((448.0-172.0) * 24.0) / 11.0
# ['Blue Goal', 'Blue Platform', 'Blue Robot', 'Neutral Goal', 'Red Goal', 'Red Platform', 'Red Robot', 'Ring']
width_dict = {"7":3.5, "3":12.5, "0":12.5, "4":12.5, "2":5.5, "5":5.5, "1":53.0, "6":53.0}

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

def obj_distance(obj, depth_frame):
    # file parsing
    obj_array = obj.split()
    x, y, w, h = int(round(float(obj_array[1]))), int(round(float(obj_array[2]))), float(obj_array[3]), float(obj_array[4])

    # calculating distance using trigonometric properties
    trig_distance = (width_dict[str(obj_array[0])] * focal_length)/w
    
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

frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())

# IO for trained YOLOv5 model (color_img->game_objects)
cv2.imwrite('img.png', color_image)

run(os.path.basename("static/best_torchscript.pt"), os.path.basename("img.png"), conf_thres=.6, name="yolo_obj", save_txt=True, nosave=True, exist_ok=True)

# reading labels from localization output
game_objects = []
with open('runs/detect/yolo_obj/labels/img.txt', 'r+') as f:
    game_objects = f.readlines()
    f.truncate(0)
    f.close()

# calculating distance for all game objects in frame
for obj in game_objects:
    print(f'{obj.split()[0]}: {obj_distance(obj, depth_frame)}')