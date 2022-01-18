import cv2 as cv
import numpy as np


def determine_platform_state(img, center, plat_width, plat_height): 
    """
    This function takes in the image and bounding bax and determines the state of the platform (left, center, right)

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
