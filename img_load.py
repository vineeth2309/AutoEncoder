import cv2
import numpy as np

img = cv2.imread('C:/Users/vinee/OneDrive/Desktop/Courses/Sem 1/Neural Computation/Semantic-Segmentation/data/training/image_2/000000_10.png')
img2 = cv2.resize(img, (512, 224))
print(img.shape)
cv2.imshow("FRAME", img)
cv2.imshow("FRAME2", img2)
cv2.waitKey()
