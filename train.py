from model import mobilenet_v1
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import optimizers
from dataloader import train_iterator
import math


def wing_loss(landmarks, labels, w=10., epsilon=2.):
    x = landmarks - labels
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    # absolute_x = torch.abs(x)
    absolute_x = tf.abs(x)
    losses = tf.where((w > absolute_x),w * tf.math.log(1.0 + absolute_x / epsilon),absolute_x - c)
    loss = tf.reduce_mean(tf.reduce_mean(losses, axis=[1]), axis=0)
    # losses = torch.mean(losses, dim=1, keepdim=True)
    # loss = torch.mean(losses)
    return loss


def train_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)

        # loss = tf.losses.MSE(labels,prediction)
        # loss = tf.reduce_mean(loss)
        loss = wing_loss(prediction,labels)

        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction

def train(model, data_iterator, optimizer):

    for i in tqdm(range(int(49063/50))):
        images, labels = data_iterator.next()
        ce, prediction = train_step(model, images, labels, optimizer)

        print('mse: {:.6f}'.format(ce))

class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
        self.warm_up_step = warm_up_step
        super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                    decay_steps=decay_steps,
                                                    alpha=alpha,
                                                    name=name)
    @tf.function
    def __call__(self, step):
        if step <= self.warm_up_step:
            return step / self.warm_up_step * self.initial_learning_rate
        else:
            return super(CosineDecayWithWarmUP, self).__call__(step - self.warm_up_step)

if __name__ == '__main__':
    train_data_iterator = train_iterator()

    model = mobilenet_v1(input_shape=[128, 128, 3])
    model.build(input_shape=(None,) + (128,128,3))

    # model = tf.keras.models.load_model("./h5/hand-198.h5")

    model.summary()

    # optimizer = optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    optimizer = optimizers.Adam()


    for epoch_num in range(200):
        train(model, train_data_iterator, optimizer)
        if epoch_num%3==0:
            model.save('./h5/hand-%s.h5'%epoch_num, save_format='h5')

