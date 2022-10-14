import os

import cv2

file = open('labelless140.txt', mode='a+')
list = os.listdir('./data/train/')
for i in range(len(list)):
    img = cv2.imread('./data/train/%s'%list[i])
    h,w,c = img.shape
    if h > 140 and w > 140:
        file.write('./dataset/label/%s.npy \n' % (list[i][:-4]))
file.close()
