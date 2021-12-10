import cv2 as cv
import numpy as np
import math

def determine_platform_state(img, center, plat_height, plat_width): 
    height = int(plat_height / 2)
    width = int(plat_width / 2)
    img = img[center[1]-height:center[1]+height, center[0]-width:center[0]+width]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 75, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 75, minLineLength=(plat_width * 0.6), maxLineGap=90)
    bins = [0,0,0]

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
    
    if bins[0] > bins[2]:
        print("Platform Left")
    elif bins[2] > bins[0]:
        print("Platform Right")
    else:
        print("Platform Center")
    #print(bins)

    # Print test image
    cv.imshow("Platform Image", img)
    cv.waitKey(0) 
    cv.destroyAllWindows()



# Testing main
def main():
    orig = cv.imread(cv.samples.findFile("platform.png"))
    determine_platform_state(orig, (370,432), 197, 291)

if __name__ == '__main__':
    main()
