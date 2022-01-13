import cv2 as cv
from platform_state import determine_platform_state


# Testing main
def main():
    # Testing values
    robot_position = (200,100) #(x,y)
    robot_rotation = 30
    orig = cv.imread(cv.samples.findFile("images/platform.png"))
    
    # Center 50% of the field
    # (Assuming 500-X field)
    # Only angles up to 30 degrees off center
    # (Assuming 180 to -180 rotation and 0-deg facing towards blue platform)
    if robot_position[0] >= 125 and robot_position <= 425:
        if abs(robot_rotation) <= 60:
            # Blue Platform State
            determine_platform_state(orig, (370,432), 291, 197)
        elif abs(robot_rotation) >= 120:
            # Red Platform State
            determine_platform_state(orig, (370,432), 291, 197)
    # Ignore Platform Image if criteria not met

if __name__ == '__main__':
    main()
