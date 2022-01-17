# senior-design-cv
Computer vision project for the MSOE Senior Design Team

### Localization Pipeline
1. Set up stream to simultaneously intake RGB images and depth maps from D435 camera
2. Identify objects in RGB image using a custom trained YOLOv5 network and Batch Mode Cluster-NMS
3. Obtain the distance from the camera to each of the identified objects using OpenCV/trig and the depth map
4. Use trig and the D435's intrinsic parameters to determine the horizontal/vertical angles of the identified objects relative to the camera
5. Make use of the ros_tf2 library to create a static broadcaster ROS node that transforms the object vectors (created from the data in steps 3 and 4) from the coordinate frame of the camera to the coordinate frame of the center of the robot, at ground level, and then publishes them
6. Create a second ROS node which subscribes to the static broadcaster ROS node which uses the coordinates of the robot (x, y, θ) and the object vectors, which are now in the robot's coordinate frame, to determine the coordinates of each object
7. Use OpenCV to determine the game state of all platform and goal objects