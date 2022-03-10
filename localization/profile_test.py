import pyrealsense2 as rs
import cv2
from detect import run, non_max_suppression
import numpy as np
import torch
from utils.plots import Annotator
import json

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
model = torch.jit.load('static/best_torchscript.pt', map_location=torch.device('cuda:0'))
model.model.float()
im = torch.zeros((1, 3, 640, 640)).to(torch.device('cuda:0')).type(torch.float)  # input image
model.forward(im)
# CPU:
# model = torch.jit.load('static/best_torchscript.pt')

def obj_distance(obj, depth_frame):
    # object parsing
    x1, y1, x2, y2, _, cls = obj

    # calculating distance using trigonometric properties
    trig_distance = (width_dict[str(int(cls))] * focal_length)/(x2-x1) 
    
    # calculating center of object
    x = (x1 + x2)/2 
    y = (y1 + y2)/2
    
    # extract average distance from depth map and convert to inches
    depth_distance_meters = depth_frame.get_distance(x, y)
    # depth_distance_meters = (depth_frame.get_distance(x, y) +\
                            #  depth_frame.get_distance(x+2, y) +\
                            #  depth_frame.get_distance(x, y+2) +\
                            #  depth_frame.get_distance(x-2, y) +\
                            #  depth_frame.get_distance(x, y-2))/5.0
    
    # weighting and combining localization methods
    distance = (trig_distance * .2) + (depth_distance_meters * .8) 
    
    # in the event that depthmap can't detect distance, only use trig distance
    if (depth_distance_meters == 0):
        distance = trig_distance
    
    return distance

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[[0, 2]] -= pad[0]  # x padding
    coords[[1, 3]] -= pad[1]  # y padding
    coords[:4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[0].clamp_(0, shape[1])  # x1
        boxes[1].clamp_(0, shape[0])  # y1
        boxes[2].clamp_(0, shape[1])  # x2
        boxes[3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[[0, 2]] = boxes[[0, 2]].clip(0, shape[1])  # x1, x2
        boxes[[1, 3]] = boxes[[1, 3]].clip(0, shape[0])  # y1, y2

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    # Model inference & NMS
    resized = cv2.resize(color_image, (640, 640), interpolation = cv2.INTER_AREA)
    model_input = torch.reshape(torch.tensor(resized).float(), (1, 3, 640, 640)).cuda()
    model_input /= 255
    results = model(model_input)
    print(results[0].shape)
    print(results[1][0].shape)
    nms_results = non_max_suppression(results, conf_thres=0.7)[0]

    annotator = Annotator(color_image.copy())

    #print(nms_results.shape)
    #print(nms_results)

    # Calculates the distance of all game objects in frame
    HFOV, VFOV = 86, 57
    # TODO calculate translation between robot and field using robots location at time of image capture
    for obj in nms_results:
        obj[:4] = scale_coords([640,640], obj[:4], (480, 640, 3)).round()

        annotator.box_label(obj[:4].cpu())

        dist = float(obj_distance(obj, depth_frame).cpu())
        x1, y1, x2, y2, conf, cls = obj.cpu()
        x = float((x1 + x2)/2.0)
        y = float((y1 + y2)/2.0)
        h_angle = np.radians(((x - 320.0)/(320.0))*(HFOV/2))
        v_angle = np.radians(((y - 320.0)/(320.0))*(VFOV/2))

        # Convert polar angles into vector
        v_x = dist * np.cos(v_angle) * np.cos(h_angle)
        v_y = dist * np.cos(v_angle) * np.sin(h_angle)
        v_z = dist * np.sin(v_angle)
        vec = np.array([v_x, v_y, v_z])
    
    im0 = annotator.result()
    cv2.imshow('image',im0)
    cv2.waitKey(0)

