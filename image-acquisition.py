import pyrealsense2 as rs
import cv2
import numpy as np

from time import time
from datetime import datetime


# Defines how long to wait between captures
image_interval_ms = 1000

# Defines how long to capture images for. Set to 0 if
# you wish to capture images indefinitely (or until quit)
capture_elapsed_ms = 0

# Defines the parent folder to save images under
save_folder = "C:\\Users\\bowsert\\Downloads\\test"


# Gets the current time in milliseconds, from epoch
def get_current_ms():
    return int(time() * 1000) 

# Defines a save directory local to the Jupyter Notebook
def get_save_directory(x):
    return f"{save_folder}/{get_unique_image_id(x)}"

# Returns a unique timestamp-based name for a file
def get_unique_image_id(x):
    date_stamp = f"img_{datetime.now()}"
    date_stamp = date_stamp.replace('-', '')
    date_stamp = date_stamp.replace(':', '')
    date_stamp = date_stamp.replace('.', '')
    date_stamp = date_stamp.replace(' ', '_')
    return date_stamp + str(x) + '.jpg'


if __name__ == "__main__":
    print(f'Beginning time lapse recording at an interval of {image_interval_ms/1000} seconds/photo')
    print('Hold [Ctrl]+[C], or [Esc] to exit')
    start_time_ms = get_current_ms()
    
    # For USB Camera
    # stream0 = cv2.VideoCapture(1)

    # For Intel Realsense Camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    while True:
        if abs(start_time_ms - get_current_ms()) >= image_interval_ms:
            # ret, frame0 = stream0.read()
            # cv2.imwrite(get_save_directory(0), frame0)

            frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imwrite(get_save_directory(0), color_image)


            start_time_ms = get_current_ms()
