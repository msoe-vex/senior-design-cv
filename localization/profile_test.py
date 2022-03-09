import pyrealsense2 as rs
import cv2
from detect import run, non_max_suppression
import numpy as np
import torch

# TODO: 
# - Find out why x1/y1 values are out of frame
# - Find out why max detections for NMS are always being reached
# - Optimize NMS code

# Calculation for object distance based on bounding box dimensions in meters
focal_length = ((448.0-172.0) * (24.0*0.0254)) / (11.0*0.0254)
# Width of game objects in meters
# ['Blue Goal', 'Blue Platform', 'Blue Robot', 'Neutral Goal', 'Red Goal', 'Red Platform', 'Red Robot', 'Ring']
width_dict = {"7":3.5*0.0254, "3":12.5*0.0254, "0":12.5*0.0254, "4":12.5*0.0254, "2":5.5*0.0254, "5":5.5*0.0254, "1":53.0*0.0254, "6":53.0*0.0254}

# Constants for object localization
HFOV, VFOV = 86, 57
# TODO calculate cameraToRobotRotation and cameraToRobotTranslation Matrices (import vector_transform code)

# Initialize pipeline and start stream
pipeline = rs.pipeline()
config = rs.config()
# For real time D435 use:
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_device_from_file('static/test-run-30-sec.bag')
pipeline.start(config)

# GPU:
# model = torch.jit.load('static/best_torchscript.pt', map_location=torch.device('cuda:0'))
# CPU:
model = torch.jit.load('static/best_torchscript.pt')

def obj_distance(obj, depth_frame):
    # object parsing
    x1, y1, x2, y2, _, cls = obj

    # calculating distance using trigonometric properties
    trig_distance = (width_dict[str(int(cls))] * focal_length)/(x2-x1) 
    
    # calculating center of object
    x = (x1 + x2)/2 
    y = (y1 + y2)/2
    
    # extract average distance from depth map and convert to inches
    depth_distance_meters = (depth_frame.get_distance(x, y) +\
                             depth_frame.get_distance(x+2, y) +\
                             depth_frame.get_distance(x, y+2) +\
                             depth_frame.get_distance(x-2, y) +\
                             depth_frame.get_distance(x, y-2))/5.0
    
    # weighting and combining localization methods
    distance = (trig_distance * .2) + (depth_distance_meters * .8) 
    
    # in the event that depthmap can't detect distance, only use trig distance
    if (depth_distance_meters == 0):
        distance = trig_distance
    
    return distance


while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    # Model inference & NMS
    resized = cv2.resize(color_image, (640, 640), interpolation = cv2.INTER_AREA)
    model_input = torch.reshape(torch.tensor(resized).float(), (1, 3, 640, 640)).cuda()
    results = model(model_input)[0]
    nms_results = non_max_suppression(results, conf_thres=0.7)[0]


    # Calculates the distance of all game objects in frame
    HFOV, VFOV = 86, 57
    print(nms_results.shape)
    # TODO calculate translation between robot and field using robots location at time of image capture
    for obj in nms_results:
        dist = obj_distance(obj, depth_frame)
        x1, y1, x2, y2, conf, cls = obj
        x = (x1 + x2)/2 
        y = (y1 + y2)/2
        h_angle = np.radians(((x - 320.0)/(320.0))*(HFOV/2))
        v_angle = np.radians(((y - 320.0)/(320.0))*(VFOV/2))

        # Convert polar angles into vector
        v_x = dist * np.cos(v_angle) * np.cos(h_angle)
        v_y = dist * np.cos(v_angle) * np.sin(h_angle)
        v_z = dist * np.sin(v_angle)
        vec = np.array([v_x, v_y, v_z])
