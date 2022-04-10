import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU


AUTOTUNE = tf.data.experimental.AUTOTUNE
DEPTH = 17

class DnCNN(Model):
    def __init__(self, depth=DEPTH):
        super(DnCNN, self).__init__()

        self.conv1 = Conv2D(96, 3, padding='same', activation='relu')

        self.conv_bn_relu = [ConvBNReLU() for i in range(depth - 2)]

        self.conv_final = Conv2D(3, 3, padding='same', kernel_initializer=he_uniform())

    def call(self, x):
        out = self.conv1(x)
        for cbr in self.conv_bn_relu:
            out = cbr(out)
        return x - self.conv_final(out)


class ConvBNReLU(Model):
    def __init__(self):
        super(ConvBNReLU, self).__init__()
        self.conv = Conv2D(96, 3, padding='same', kernel_initializer=he_uniform())
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

def augment(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, 3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


def configure(ds):
    ds = ds.cache()
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(1)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def doDenoise(destination, mode, target):

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ds = tf.data.Dataset.list_files(destination)
    ds = configure(ds)

    model = DnCNN()

    if mode == 'l15':
        model.load_weights('models/l15/l15')
    else:
        model.load_weights('models/l25/l25')

    image_batch = next(iter(ds))

    predictions = model(image_batch, training=False)

    out = Image.fromarray(np.uint8(predictions[0].numpy() * 255))

    destination = "/".join([target, 'temp.jpg'])
    if os.path.isfile(destination):
        os.remove(destination)
    out.save(destination)
