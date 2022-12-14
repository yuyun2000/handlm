import tensorflow as tf
import numpy as np
import time
import json
import os
import cv2

# def load_list(list_path='./labelv1.txt', image_root_path='./dataset/handpose_datasets_v1/'):
#   images = []
#   labels = []
#   with open(list_path, 'r') as f:
#     for line in f:
#       # print(line)
#       images.append(os.path.join(image_root_path, line[16:-6] + '.jpg'))
#       labels.append(line[:-2])
#   return images, labels
# img,_ = load_list()
# def representative_dataset_gen():
#     for i in range(len(img)):
#       image = cv2.imread(img[i])
#       image = cv2.resize(image,(128,128))
#       image = image.astype(np.float32).reshape(1, 128, 128, 3)
#       image = image / 255
#       yield [image]
# # 装载预训练模型
# model = tf.keras.models.load_model("./handlm.h5")
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# # 设置优化器
# converter._experimental_disable_per_channel = True
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # 获得标注数据
# converter.representative_dataset = representative_dataset_gen
# # 执行转化操作
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
#
# tflite_quant_model = converter.convert()
# open("ministall.tflite", "wb").write(tflite_quant_model)

#####################推理

def run_tflite_model(tflite_file, test_image):

  # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Check if the input type is quantized, then rescale input data to uint8

    if input_details['dtype'] == np.int8:
      print("int8")
      input_scale, input_zero_point = input_details["quantization"]
      # print(input_scale,input_zero_point)
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    # print(test_image)
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    return output
# print(x[0].reshape(28,28))
# print(x[1].shape)

img = cv2.imread('./data/test/1.jpg')
img0 = cv2.resize(img,(128,128))
# img = np.array(img,dtype=np.int8)
# img = img-128
print(img)
img = img0/255
out = run_tflite_model("handlm.tflite",img)
out = np.array(out,dtype=np.int32)
print(out)
out = out+118
out = out* 0.003795162308961153
# out = out*128
print(out)

s = 128
out = np.array(out).reshape(21,2)*s
for i in range(21):
    x = int(out[i][0])
    y = int(out[i][1])
    print(x,y)
    cv2.circle(img0, (x, y), 1, (0, 0, 255), 2)
img0 = np.array(img0,dtype=np.uint8)
img0 = cv2.resize(img0,(256,256))
cv2.imshow('1',img0)
cv2.waitKey(0)
cv2.destroyAllWindows()