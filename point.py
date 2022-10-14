import cv2
import numpy as np

label = np.load('./data/train/label/1662368151838.npy')
img = cv2.imread('./data/train/img/1662368151838.jpg')
for i  in range(21):
    cv2.circle(img,(int(label[i][0]), int(label[i][1])),2, (0, 0, 255), thickness=2)
cv2.imshow('1',img)
cv2.waitKey(0)