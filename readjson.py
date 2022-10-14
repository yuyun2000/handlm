import json
import numpy as np
import cv2
result = open("./dataset/handpose_datasets_v2/2022-04-15_05-13-16_046322.json","r",encoding="utf-8")
point = json.load(result)
# print(point['info'][0]['pts']['%s'%0])
# point = np.array(point)
# print(point)
# print(point[0])


img = cv2.imread('./dataset/handpose_datasets_v2/2022-04-15_05-13-16_046322.jpg')
h,w,c = img.shape
for i  in range(21):
    x = int(point['info'][0]['pts']['%s'%i]['x'])
    y = int(point['info'][0]['pts']['%s'%i]['y'])
    print(x,y)
    cv2.circle(img,(x,y ),1, (0, 0, 255), thickness=1)
cv2.imshow('1',img)
cv2.waitKey(0)