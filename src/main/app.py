import os
import cv2
from utils import DetectStripesData

def read_image(img_path):
  assert os.path.exists(img_path), 'Path does not exist: {}'.format(img_path)
  img = cv2.imread(img_path)
  return img

def main():
  img_path = "/Users/vinodpatil/LAB/practice/PycharmProjects/lumber_scanner/src/main/resources/Lumber_Stripes.JPG"
  img = read_image(img_path)

  obj = DetectStripesData.DetectStripesData()
  obj.findStripes(img)

if __name__ == '__main__':
  main()

