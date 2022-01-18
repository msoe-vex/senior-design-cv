import cv2 as cv
import numpy as np

def platform_state_with_determinism(robot_position, robot_rotation, img, center, plat_width, plat_height):
    """
    This function takes in the image, bounding box, and robot position.
    Determines the state of the platform (left, center, right) and which color platform (blue, red)

    :param robot_position: (x,y) position of the robot (Assuming 500-X field)
    :param robot_rotation: Angle of the robot (Assuming 180 to -180 rotation and 0-deg facing towards blue platform)
    :param img: Full images from camera
    :param center: center of bounding box (x,y)
    :param plat_width: width of platform bounding box
    :param plat_height: height of platform bounding box
    :return: integer tuple (color, state)
             color: (-1,0,1) -> (unknown, blue, red)
             state: (-1,0,1,2) -> (unknown, left, center, right) 
    """ 
    # Center 50% of the field
    # Only angles up to 30 degrees off center
    if robot_position[0] >= 125 and robot_position <= 425:
        if abs(robot_rotation) <= 60:
            # Blue Platform State
            plat_state = determine_platform_state(img, center, plat_width, plat_height)
            return 0,plat_state
        elif abs(robot_rotation) >= 120:
            # Red Platform State
            plat_state = determine_platform_state(img, center, plat_width, plat_height)
            return 1,plat_state
    return -1,-1


def determine_platform_state(img, center, plat_width, plat_height): 
    """
    This function takes in the image and bounding box and determines the state of the platform (left, center, right)

    :param img: Full images from camera
    :param center: center of bounding box (x,y)
    :param plat_width: width of platform bounding box
    :param plat_height: height of platform bounding box
    :return: returns -1 if state of platform cannot be determined
             returns one integer (0,1,2) corresponding to (left, center, right) state
    """ 
    # Resize image to only located platform object
    height = int(plat_height / 2)
    width = int(plat_width / 2)
    img = img[center[1]-height:center[1]+height, center[0]-width:center[0]+width]
    # TODO: verify center is returned from model as (x,y) instead of (y,x)

    # Transform image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 75, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 75, minLineLength=(plat_width * 0.6), maxLineGap=90)
    bins = [0,0,0]

    # Bin line slopes
    if lines.size == 0:
        # State Unknown
        return -1
    else:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2-y1)/(x2-x1)
            if slope <= -0.3:
                bins[0] += 1
            elif slope >= 0.3:
                bins[2] += 1
            else:
                bins[1] += 1
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 128), 1)
        
        # # Print test image
        # cv.imshow("Platform Image", img)
        # cv.waitKey(0) 
        # cv.destroyAllWindows()

        # Determine direction of platform based on bins
        if bins[0] > bins[2]:
            # Platform Left
            return 0
        elif bins[2] > bins[0]:
            # Platform Right
            return 2
        else:
            # Platform Center
            return 1

    



# Testing main
def main():
    orig = cv.imread(cv.samples.findFile("images/platform.png"))
    determine_platform_state(orig, (370,432), 291, 197)

if __name__ == '__main__':
    main()
