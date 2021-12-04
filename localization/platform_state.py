import cv2 as cv
import numpy as np
import math

def determine_platform_state(img, center, plat_height, plat_width): 
    height = int(plat_height / 2)
    width = int(plat_width / 2)
    img = img[center[1]-height:center[1]+height, center[0]-width:center[0]+width]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 75, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dist = math.dist([x1,y1], [x2,y2])
        if dist >= plat_width * 0.6:
            new_lines.append(line[0])
        print(dist)
        print(line)
    print(new_lines)
    print(plat_width * 0.6)

    for line in new_lines:
        x1, y1, x2, y2 = line
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 128), 1)
    

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
