import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers

# （1）标准卷积模块
def conv_block(input_tensor, filters, alpha, kernel_size=(3, 3), strides=(1, 1)):
    # 超参数alpha控制卷积核个数
    filters = int(filters * alpha)

    # 卷积+批标准化+激活函数
    x = layers.Conv2D(filters, kernel_size,
                      strides=strides,  # 步长
                      padding='same',  # 0填充，卷积后特征图size不变
                      use_bias=False)(input_tensor)  # 有BN层就不需要计算偏置
    x = layers.BatchNormalization()(x)  # 批标准化
    x = layers.ReLU(6.0)(x)  # relu6激活函数
    return x  # 返回一次标准卷积后的结果

# （2）深度可分离卷积块
def depthwise_conv_block(input_tensor, point_filters, alpha, depth_multiplier, strides=(1, 1)):
    # 超参数alpha控制逐点卷积的卷积核个数
    point_filters = int(point_filters * alpha)

    # ① 深度卷积--输出特征图个数和输入特征图的通道数相同
    x = layers.DepthwiseConv2D(kernel_size=(3, 3),  # 卷积核size默认3*3
                               strides=strides,  # 步长
                               padding='same',  # strides=1时，卷积过程中特征图size不变
                               depth_multiplier=depth_multiplier,  # 超参数，控制卷积层中间输出特征图的长宽
                               use_bias=False)(input_tensor)  # 有BN层就不需要偏置

    x = layers.BatchNormalization()(x)  # 批标准化

    x = layers.ReLU(6.0)(x)  # relu6激活函数

    # ② 逐点卷积--1*1标准卷积
    x = layers.Conv2D(point_filters, kernel_size=(1, 1),  # 卷积核默认1*1
                      padding='same',  # 卷积过程中特征图size不变
                      strides=(1, 1),  # 步长为1，对特征图上每个像素点卷积
                      use_bias=False)(x)  # 有BN层，不需要偏置

    x = layers.BatchNormalization()(x)  # 批标准化

    x = layers.ReLU(6.0)(x)  # 激活函数

    return x  # 返回深度可分离卷积结果

def conv_block_withoutrelu(
        inputs,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1)
):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(
        inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def fireblock(inputs,indim,middim,outdim):
    x1 = conv_block(inputs,middim,1,(1,1))

    x2 = conv_block(x1,outdim,1,kernel_size=(1,1))
    if outdim>=128:
        x3 = depthwise_conv_block(x1,outdim,1,1)
    else:
        x3 = conv_block(x1,outdim,1)
    out = tf.concat((x2,x3),3)
    return out


import numpy as np
def process_layer(image):
    np.random.seed(2022)#不同的顺序会影响最终的结果，设定随机的顺序减少这些影响
    mode = np.random.randint(3)
    if mode ==0:
        image = tf.image.random_brightness(image,max_delta=0.125)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif mode == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=0.125)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif mode == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=0.125)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif mode == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=0.125)

    return tf.clip_by_value(image,0.0,1.0)#把最终的结果限制在0-1的区间

def squeezenet( input_shape,classes=42):
    # 创建输入层
    inputs = layers.Input(shape=input_shape)
    inputs = process_layer(inputs)
    x = conv_block(inputs, 32, 1, strides=(2, 2))  # 步长为2，压缩宽高，提升通道数
    x = conv_block(x, 64, 1,strides=(2, 2))

    x = fireblock(x,64,16,64)#输入64，压缩为16，输出64*2
    x = fireblock(x,128,16,64)
    x = layers.MaxPool2D(pool_size=(2, 2),strides=2)(x)

    x = fireblock(x,128,32,128)#输入64，压缩为16，输出64*2
    x = fireblock(x,256,32,128)
    x = layers.MaxPool2D(pool_size=(2, 2),strides=2)(x)
    x = fireblock(x, 256, 48, 192)
    x = fireblock(x, 384, 48, 192)
    x = fireblock(x, 384, 64, 256)
    x = fireblock(x, 512, 64, 256)

    x = layers.Dropout(rate=0.5)(x)
    x = conv_block(x, 42, 1, (1, 1))
    x = layers.GlobalAveragePooling2D()(x)  # 通道维度上对size维度求平均

    # 构建模型
    model = Model(inputs, x)
    # 返回模型结构
    return model



from Flops import try_count_flops
if __name__ == '__main__':
    # 获得模型结构
    model = squeezenet(input_shape=[128, 128,3],classes=42)
    flops = try_count_flops(model)
    print(flops/1000000)
    # # 查看网络模型结构
    model.summary()
    model.save("./mbtest.h5", save_format="h5")
    # print(model.layers[-3])

    # model = tf.keras.models.load_model("./mbtest.h5")
    # model.summary()