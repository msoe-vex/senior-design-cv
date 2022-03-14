import pyrealsense2 as rs
import cv2
import numpy as np
import torch

from utils.plots import Annotator
from models.experimental import attempt_load
from utils.general import non_max_suppression

# TODO: 
# - Find out why x1/y1 values are out of frame

# Calculation for object distance based on bounding box dimensions in meters
focal_length = ((448.0-172.0) * (24.0*0.0254)) / (11.0*0.0254)
# Width of game objects in meters
labels = {0:'Blue Goal', 1:'Blue Robot', 2:'Neutral Goal', 3:'Platform', 4:'Red Goal', 5:'Red Robot', 6:'Ring'}
width_dict = {"6":3.5*0.0254, "0":12.5*0.0254, "2":12.5*0.0254, "4":12.5*0.0254, "1":5.5*0.0254, "5":5.5*0.0254, "3":53.0*0.0254}

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
model = attempt_load('static/best.pt', map_location=torch.device('cuda:0'))
model.model.half()
img = torch.zeros((1, 3, 480, 640)).to(torch.device('cuda:0')).type(torch.half)
model.forward(img)


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

    # Format Image and put on GPU
    color_image = np.ascontiguousarray(color_frame.get_data()).transpose((2, 0, 1))
    torch.cuda.synchronize()
    model_input = torch.from_numpy(color_image).cuda().half()
    model_input /= 255
    model_input = model_input[None]
    torch.cuda.synchronize()

    # Run model inference
    results = model(model_input)[0]
    torch.cuda.synchronize()

    # Run NMS algorithm
    nms_results = non_max_suppression(results, conf_thres=0.5)[0]
    torch.cuda.synchronize()
    print(nms_results)

    # Set up annotator to get test output image (TODO remove - only for testing)
    img = cv2.cvtColor(np.ascontiguousarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)
    annotator = Annotator(img)


    # Calculates the distance of all game objects in frame
    HFOV, VFOV = 86, 57
    # TODO calculate translation between robot and field using robots location at time of image capture
    for obj in nms_results:

        annotator.box_label(obj[:4].cpu())

        dist = float(obj_distance(obj, depth_frame).cpu())
        x1, y1, x2, y2, conf, cls = obj.cpu()
        x = float((x1 + x2)/2.0)
        y = float((y1 + y2)/2.0)
        h_angle = np.radians(((x - 320.0)/(320.0))*(HFOV/2)) #TODO - verfiy
        v_angle = np.radians(((y - 320.0)/(320.0))*(VFOV/2)) #TODO - verfiy

        # Convert polar angles into vector
        v_x = dist * np.cos(v_angle) * np.cos(h_angle)
        v_y = dist * np.cos(v_angle) * np.sin(h_angle)
        v_z = dist * np.sin(v_angle)
        vec = np.array([v_x, v_y, v_z])
        print(vec)
    
    
    # Print test output (TODO remove - only for testing)
    im0 = annotator.result()
    cv2.imshow('image',im0)
    cv2.waitKey(0)

