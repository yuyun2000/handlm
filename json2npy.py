import json
import numpy as np
import os
list1 = os.listdir('./dataset/handpose_datasets_v2')
print(len(list1))
for i in range(len(list1)):
    if 'json' in list1[i]:
        # print(list1[i][:-5])
        result = open("./dataset/handpose_datasets_v2/%s"%list1[i], "r", encoding="utf-8")
        point = json.load(result)
        result.close()
        label = np.zeros((21, 2))
        for j in range(21):
            label[j][0] = int(point['info'][0]['pts']['%s' % j]['x'])
            label[j][1] = int(point['info'][0]['pts']['%s' % j]['y'])
        np.save('./dataset/label/%s'%list1[i][:-5],label)


