import cv2 as cv
import numpy as np
from math import atan2, cos, sin, sqrt, pi
import time

def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

def getOrientationDraw(pts, img):
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
 
  # Draw the principal components
  cntr = (int(mean[0,0]), int(mean[0,1]))
  cv.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
  drawAxis(img, cntr, p1, (255, 255, 0), 1)
  drawAxis(img, cntr, p2, (0, 0, 255), 5)
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0])

  # Label with the rotation angle
  label = str(-int(np.rad2deg(angle))) + " degrees"
  textbox = cv.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
  cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)

def getOrientation(pts, img):
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, _ = cv.PCACompute2(data_pts, mean)

  # Return angle
  return atan2(eigenvectors[0,1], eigenvectors[0,0])

def is_goal_tipped(img, center, plat_width, plat_height): 
  # Resize image to only located goal object
  height = int(plat_height / 2)+5
  width = int(plat_width / 2)+5
  img = img[center[1]-height:center[1]+height, center[0]-width:center[0]+width]
  
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
  contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
  angles = []

  for i, c in enumerate(contours):
      # Calculate the area of each contour
      area = cv.contourArea(c)
      # Ignore contours that are too small or too large
      if area < 2000 or 100000 < area:
          continue
      # cv.drawContours(img, contours, i, (0, 0, 255), 2)
      # Find orientation of shape
      angles.append(abs(int(np.rad2deg(getOrientation(c, img)))))
  
  # # Print test image
  # cv.imshow("Goal Image", img)
  # cv.waitKey(0) 
  # cv.destroyAllWindows()

  # Determine goal state and return result
  if len(angles) != 1:
    return -1
  elif angles[0] > 45:
    return 1
  else:
    return 0


# Testing main
def main():
  orig = cv.imread(cv.samples.findFile("images/g-eb3.jpg"))
  print(is_goal_tipped(orig, (149,346), 87, 49)) #yellow untipped: g-eb3.jpg
  print(is_goal_tipped(orig, (594,379), 90, 82)) #blue on platform: g-eb3.jpg
  orig = cv.imread(cv.samples.findFile("images/g-507.jpg"))
  print(is_goal_tipped(orig, (397,379), 96, 168)) #yellow tipped closer: g-507.jpg
  print(is_goal_tipped(orig, (306,331), 66, 66)) #yellow tipped further: g-507.jpg


if __name__ == '__main__':
    main()
