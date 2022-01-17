# senior-design-cv
Computer vision project for the MSOE Senior Design Team

### Localization Pipeline
1. Set up stream to simultaneously intake RGB images and depth maps from an (Intel RealSense D435 Depth Camera)[https://www.intelrealsense.com/depth-camera-d435/]
2. Identify objects in RGB image using a custom trained [YOLOv5](https://github.com/ultralytics/yolov5) network and [Batch Mode Cluster-NMS](https://github.com/Zzh-tju/yolov5), and then use OpenCV to determine the state of all platform and goal objects
3. Obtain the distance from the camera to each of the identified objects using OpenCV/trig and the depth map
4. Use the [D435's intrinsic parameters](https://dev.intelrealsense.com/docs/projection-in-intel-realsense-sdk-20) and trig to determine the horizontal/vertical angles of the identified objects relative to the camera
5. Make use of the [ros_tf2 library](http://wiki.ros.org/tf2) to create a [static broadcaster ROS node](http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20static%20broadcaster%20%28Python%29) that transforms the object vectors (created from the data in steps 3 and 4) from the coordinate frame of the camera (1) to the coordinate frame of the center of the robot, at ground level (2)
![image](https://user-images.githubusercontent.com/43486503/149828054-9dfc3dec-5e5d-4e76-8038-454067157f82.png)
6. Create a second ROS node which subscribes to the static broadcaster ROS node which uses the coordinates of the robot (x, y, θ) and the object vectors, which are now in the robot's coordinate frame, to determine the coordinates of each object
7. Use second ROS node to publish the objects game states and coordinates
