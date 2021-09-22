import cv2
import os

from os import path
from time import time
from datetime import datetime
import sys


# Defines how long to wait between captures
image_interval_ms = 1000

# Defines how long to capture images for. Set to 0 if
# you wish to capture images indefinitely (or until quit)
capture_elapsed_ms = 0

# Defines the parent folder to save images under
save_folder = "D:\Run1"


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
    stream_links = [1,2]
    
    stream0 = cv2.VideoCapture(1)
    stream1 = cv2.VideoCapture(2)
    while True:
        if abs(start_time_ms - get_current_ms()) >= image_interval_ms:
            ret, frame0 = stream0.read()
            ret, frame1 = stream1.read()
            cv2.imwrite(get_save_directory(0), frame0)
            cv2.imwrite(get_save_directory(1), frame1)
            start_time_ms = get_current_ms()
