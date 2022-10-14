import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75)
#
img = cv2.imread('./dataset/rgb/00081244.jpg')
h,w,c = img.shape
# img = cv2.flip(img, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res = hands.process(img)
label = np.zeros((21,2))
for i in range(21):
        cv2.circle(img, (int(w*res.multi_hand_landmarks[0].landmark[i].x), int(h*res.multi_hand_landmarks[0].landmark[i].y)), 1, (0, 0, 255),thickness=2)
        # cv2.putText(img,'%s'%i,(int(w*res.multi_hand_landmarks[0].landmark[i].x), int(h*res.multi_hand_landmarks[0].landmark[i].y)),1,10,(0,0,255),10)
        label[i][0] = int(w*res.multi_hand_landmarks[0].landmark[i].x)
        label[i][1] = int(h*res.multi_hand_landmarks[0].landmark[i].y)
img = cv2.resize(img,(256,256))
cv2.imshow('1',img)
cv2.waitKey(0)

# import os
# list1 = os.listdir('./data/train/img')
# for p in range(len(list1)):
#         img = cv2.imread('./data/train/img/%s'%list1[p])
#         h, w, c = img.shape
#         # img = cv2.flip(img, 1)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         res = hands.process(img)
#         if res.multi_handedness == None:
#                 continue
#
#         label = np.zeros((21, 2))
#         for i in range(21):
#                 label[i][0] = int(w * res.multi_hand_landmarks[0].landmark[i].x)
#                 label[i][1] = int(h * res.multi_hand_landmarks[0].landmark[i].y)
#         np.save('./data/train/label/%s'%list1[p][:-4],label)
