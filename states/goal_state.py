import cv2 as cv
import numpy as np
import math

def determine_goal_state(img, center, plat_width, plat_height): 
    height = int(plat_height / 2)
    width = int(plat_width / 2)
    img = img[center[1]-height:center[1]+height, center[0]-width:center[0]+width]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 75, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 75, minLineLength=(plat_width * 0.6), maxLineGap=90)
    

    # Print test image
    cv.imshow("Platform Image", img)
    cv.waitKey(0) 
    cv.destroyAllWindows()

# Testing main
def main():
    orig = cv.imread(cv.samples.findFile("images/g-eb3.png"))
    determine_goal_state(orig, (149,346), 87, 49) #yellow untipped: g-eb3


if __name__ == '__main__':
    main()
