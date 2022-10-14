import tensorflow as tf
import numpy as np
import os
import cv2
import torchlm

transform = torchlm.LandmarksCompose([
    torchlm.LandmarksRandomScale(prob=0.5),
    torchlm.LandmarksRandomMask(prob=0.5),
    torchlm.LandmarksRandomBlur(kernel_range=(5, 25), prob=0.5),
    torchlm.LandmarksRandomBrightness(prob=0.),
    torchlm.LandmarksRandomRotate(40, prob=0.5, bins=8),
    torchlm.LandmarksRandomCenterCrop((0.5, 1.0), (0.5, 1.0), prob=0.5)
])

def makeGaussian(size, fwhm = 2, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def load_list(list_path='./labelv1.txt',image_root_path='./dataset/handpose_datasets_v1/'):
    images = []
    labels = []
    with open(list_path, 'r') as f:
        for line in f:
            # print(line)
            images.append(os.path.join(image_root_path, line[16:-6]+'.jpg'))
            labels.append(line[:-2])
    return images, labels

def load_image(image_path, label_path):

    # print(image_path.numpy().decode())
    image = cv2.imread(image_path.numpy().decode())
    preds = np.load(label_path.numpy().decode()).reshape(21,2)

    image,preds = transform(image,preds)
    image=image.astype(np.float32)
    preds=preds.astype(np.float32)
    h,w,c = image.shape
    image = cv2.resize(image,(128,128))
    image = image / 255
    preds[:,0:1] = preds[:,0:1]/w
    preds[:,1:2] = preds[:,1:2]/h
    preds = preds.reshape(42)


    return image, preds


def train_iterator():
    images, labels = load_list()
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(len(images))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)#.cache('/mnt/ssd0.5T/train/face68.TF-data')
    dataset = dataset.repeat()
    dataset = dataset.batch(50).prefetch(1)
    it = dataset.__iter__()
    return it

if __name__ == '__main__':
    it = train_iterator()
    images, labels = it.next()
    print(labels[0])




