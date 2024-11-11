import os
import imutils
import cv2
import numpy as np


class DetectStripesData:

  def __init__(self):
    self.class_properties = {
      "blur": {
        "kernal": [7, 7],
        "sigmaX": 0
      },
      "edge": {
        "canny": {
          "tLower": 20,
          "tUpper": 150,
          "apertureSize" : 3,
          "l2Gradient" : True
        }
      },
      "dilate" : {
        "Kernel" : [7, 7],
        "iterations" : 1,
      },
      "erode" : {
        "Kernel" : [7, 7],
        "iterations" : 1,
      },
      "minArea": 100,
      "boundingBox" : {
        "color" : [0, 255, 0],
        "thickness" : 3
      },
      "Common" : {
        "resizeKernal" : [512, 512]
      }
    }

    self.count = 0

  def showImage(self, img, img_alt="img"):
    cv2.imshow(img_alt, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  def get_cl(self, count):
    cl = [
      (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
      (255, 127, 80), (100, 149, 237)
    ]
    return cl[count % len(cl)]

  def draw_rects(self, img, rectPoints):
    for rect in rectPoints:
      cv2.rectangle(img, rect[0], rect[1],
                    self.class_properties["boundingBox"]["color"],
                    self.class_properties["boundingBox"]["thickness"])
      self.showImage(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), "rects")

  def getRect(self, points):
    reshaped_points = points.reshape(-1, 2)
    min_x = int(np.min(reshaped_points[:, 0]))
    min_y = int(np.min(reshaped_points[:, 1]))
    max_x = int(np.max(reshaped_points[:, 0]))
    max_y = int(np.max(reshaped_points[:, 1]))
    return [[min_x, min_y], [max_x, max_y]]

  def findStripes(self, img: np.ndarray):
    self.showImage(img, f"original image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    self.showImage(gray, f"gray image")
    blurred = cv2.GaussianBlur(gray, self.class_properties["blur"]["kernal"], self.class_properties["blur"]["sigmaX"])
    self.showImage(blurred, f"blurred image")
    edges = cv2.Canny(blurred, self.class_properties["edge"]["canny"]["tLower"],
                      self.class_properties["edge"]["canny"]["tUpper"])
    self.showImage(edges, f"edge image")
    dilated = cv2.dilate(edges.copy(), np.ones(self.class_properties["dilate"]["Kernel"], np.uint8), iterations=self.class_properties["dilate"]["iterations"])
    self.showImage(dilated, f"dilated image")
    # eroded = cv2.erode(dilated, np.ones(self.class_properties["erode"]["Kernel"], np.uint8), iterations=self.class_properties["erode"]["iterations"])
    # self.showImage(eroded, f"eroded image")
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    minArea = self.class_properties["minArea"]

    img_copy = img.copy()
    obj_rects = []
    for c in cnts:
      area = cv2.contourArea(c)
      if area >= minArea:
        obj_rects.append(self.getRect(c.copy()))

        cv2.drawContours(img_copy, [c], -1, self.get_cl(self.count), 1)
        self.count += 1

    self.draw_rects(img, obj_rects)