import numpy
import tensorflow as tf
import cv2
import numpy as np
from utils import draw_bd_handpose
from Flops import try_count_flops

model = tf.keras.models.load_model("./h5/hand-20.h5")
# model.summary()
# flops = try_count_flops(model)
# print(flops)
# imagegt = cv2.imread('./data/test/7.jpg')
# s = 128
# imagegt = cv2.resize(imagegt,(s,s))
# image = imagegt.astype(np.float32)
# img = image / 255
# img = img.reshape(1,s,s,3)
# out = model(img,training=False)
# print(out)
# out = np.array(tf.reshape(out[0:1,:],(21,2)))*s
#
# draw_bd_handpose(imagegt,out)
# # for i in range(21):
# #     x = int(out[i][0])
# #     y = int(out[i][1])
# #     print(x,y)
# #     cv2.circle(imagegt, (x, y), 1, (0, 0, 255), 2)
# imagegt = cv2.resize(imagegt,(256,256))
# cv2.imshow('1',imagegt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#### ----------------------video

import time
t1 = time.time()
s = 128
vid = cv2.VideoCapture('./data/test/0.mp4')
fourcc = cv2.VideoWriter_fourcc(*'I420')
outv = cv2.VideoWriter('output.avi',fourcc,20,(s,s))
while True:
    flag,img = vid.read(0)
    if not flag:
        break
    imagegt = cv2.resize(img, (s, s))
    image = imagegt.astype(np.float32)
    img = image / 255
    img = img.reshape(1, s, s, 3)
    out = model(img, training=False)
    out = np.array(tf.reshape(out[0:1, :], (21, 2)))*s
    draw_bd_handpose(imagegt,out)
    for i in range(21):
        x = int(out[i][0])
        y = int(out[i][1])
        cv2.circle(imagegt, (x, y), 1, (0, 0, 255), 2)

    img0 = cv2.resize(imagegt, (s, s))
    outv.write(img0)
    cv2.imshow('1', img0)
    if ord('q') == cv2.waitKey(1):
        break
vid.release()
outv.release()
#销毁所有的数据
cv2.destroyAllWindows()

t2 = time.time()
print(t2-t1)
#
