# Functions that support both UNets.

import os
import tensorflow.compat.v1 as tf
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
import numpy as np
import glob
from keras import backend as K
from keras.models import Model
import random
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.disable_v2_behavior()

# Scheduler, currently just a stolen one as I don't know what I should be going for.
def schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# Loss function
def nrmse(y_true, y_pred):
    denom = K.sqrt(K.mean(K.square(y_true), axis=(1, 2, 3)))
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=(1, 2, 3))) / denom


# IFFT layer, used in u_net.
def ifft_layer(kspace):
    real = layers.Lambda(lambda kspace: kspace[:, :, :, 0])(kspace)
    imag = layers.Lambda(lambda kspace: kspace[:, :, :, 1])(kspace)
    kspace_complex = tf.complex(real, imag)
    rec1 = tf.abs(tf.ifft2d(kspace_complex))
    rec1 = tf.expand_dims(rec1, -1)
    return rec1


# Upgraded version, returns fewer arrays but with a faster and more efficient method.
# Still need to figure out the imaginary image domain stuff.
def get_brains(set, ADDR):

    # Note: In train, one file is (174, 256, 256).
    kspace_files_train = np.asarray(glob.glob(str(ADDR / set["addrs"]["TRAIN_ADDR"])))
    kspace_files_val = np.asarray(glob.glob(str(ADDR / set["addrs"]["VAL_ADDR"])))
    kspace_files_test = np.asarray(glob.glob(str(ADDR / set["addrs"]["TEST_ADDR"])))

    logging.info("train scans: " + str(len(kspace_files_train)))
    logging.info("val scans: " + str(len(kspace_files_val)))
    logging.info("test scans: " + str(len(kspace_files_test)))
    logging.debug("Scans loaded")

    samp_mask = np.load(str(ADDR / set["addrs"]["MASK_ADDR"]))
    shape = (256, 256)
    norm = np.sqrt(shape[0] * shape[1])

    image_train = []
    kspace_train = []
    count = 0
    for i in random.sample(range(0, len(kspace_files_train)), len(kspace_files_train)):
        brain = np.load(kspace_files_train[i]) / norm
        if count >= set["params"]["NUM_TRAIN"]:
            break
        for i in random.sample(range(0, brain.shape[0]), brain.shape[0]):
            image_train.append(
                np.abs(np.fft.ifft2(brain[i, :, :, 0] + 1j * brain[i, :, :, 1])).astype(
                    np.float64
                )
            )
            brain[i, samp_mask, :] = 0
            kspace_train.append(brain[i].astype(np.float64))
            count += 1
            if count >= set["params"]["NUM_TRAIN"]:
                break
    kspace_train = np.asarray(kspace_train)
    logging.info("kspace train: " + str(kspace_train.shape))
    image_train = np.asarray(image_train)
    image_train = np.expand_dims(image_train, axis=3)
    logging.info("image train: " + str(image_train.shape))

    image_val = []
    kspace_val = []
    count = 0
    for i in random.sample(range(0, len(kspace_files_val)), len(kspace_files_val)):
        brain = np.load(kspace_files_val[i]) / norm
        if count >= set["params"]["NUM_VAL"]:
            break
        for i in random.sample(range(0, brain.shape[0]), brain.shape[0]):
            image_val.append(
                np.abs(np.fft.ifft2(brain[i, :, :, 0] + 1j * brain[i, :, :, 1])).astype(
                    np.float64
                )
            )
            brain[i, samp_mask, :] = 0
            kspace_val.append(brain[i].astype(np.float64))
            count += 1
            if count >= set["params"]["NUM_VAL"]:
                break
    kspace_val = np.asarray(kspace_val)
    logging.info("kspace val: " + str(kspace_val.shape))
    image_val = np.asarray(image_val)
    image_val = np.expand_dims(image_val, axis=3)
    logging.info("image val: " + str(image_val.shape))

    image_test = []
    kspace_test = []
    count = 0
    for i in random.sample(range(0, len(kspace_files_test)), len(kspace_files_test)):
        brain = np.load(kspace_files_test[i]) / norm
        if count >= set["params"]["NUM_TEST"]:
            break
        for i in random.sample(range(0, brain.shape[0]), brain.shape[0]):
            image_test.append(
                np.abs(np.fft.ifft2(brain[i, :, :, 0] + 1j * brain[i, :, :, 1])).astype(
                    np.float64
                )
            )
            brain[i, samp_mask, :] = 0
            kspace_test.append(brain[i].astype(np.float64))
            count += 1
            if count >= set["params"]["NUM_TEST"]:
                break
    kspace_test = np.asarray(kspace_test)
    logging.info("kspace test: " + str(kspace_test.shape))
    image_test = np.asarray(image_test)
    image_test = np.expand_dims(image_test, axis=3)
    logging.info("image test: " + str(image_test.shape))

    logging.debug("Scans formatted")

    # Question: image_train seems to be the reconstructed images here, but it's imaginary. Why is image domain imaginary?
    # Especially since the WNet returns a real image. Does the imaginary part in image domain yield any valuable information?
    # For the time being, I'm just going to load the stats manually.
    stats = np.load(str(ADDR / set["addrs"]["STATS_ADDR"]))

    """# save k-space and image domain stats
    stats = np.zeros(4)
    stats[0] = kspace_train.mean()
    stats[1] = kspace_train.std()
    aux = np.abs(image_train[:,:,:,0] +1j*image_train[:,:,:,1])
    stats[2] = aux.mean()
    stats[3] = aux.std()
    np.save(STATS_ADDR, stats)"""

    return (
        stats,
        kspace_train,
        image_train,
        kspace_val,
        image_val,
        kspace_test,
        image_test,
    )


# Custom complex convolution.
# I feel like my understanding of his might be off. How are the amount of output filters related to the two channel
# number? Are the imaginary numbers preserved?
# Uses algebra below. I've used "|" to denote a two channel array, and "f" to denote a variable that is a part of a filter.

# (R | I) * (Rf | If) = Or | Oi = (R * Rf - I * If) | (I * Rf + R * If)

# Config function added to allow loading and saving.
class CompConv2D(layers.Layer):
    def __init__(self, out_channels, kshape=(3, 3), **kwargs):
        super(CompConv2D, self).__init__()
        self.convreal = layers.Conv2D(
            out_channels, kshape, activation="relu", padding="same"
        )
        self.convimag = layers.Conv2D(
            out_channels, kshape, activation="relu", padding="same"
        )

    def call(self, input_tensor, training=False):
        ureal, uimag = tf.split(input_tensor, num_or_size_splits=2, axis=3)
        oreal = self.convreal(ureal) - self.convimag(uimag)
        oimag = self.convimag(ureal) + self.convreal(uimag)
        x = tf.concat([oreal, oimag], axis=3)
        return x

    def get_config(self):

        config = super(CompConv2D, self).get_config()
        config.update(
            {
                "convreal": self.convreal,
                "convimag": self.convimag,
            }
        )
        return config


# U-Net model. Includes kspace domain U-Net and IFFT.
# Note: Filters are halved to maintain structure, but they probably shouldn't be. Although this may reduce performance, I think it's probably fine for testing.
# A variable (MOD) was added to make testing easier. At small scale, MOD = 1 worked best.
# Question: Can a purely kspace model be trained, with no reference to image domain? For this code, I've assumed no (also just for convenience viewing results).
# I'm pretty sure it can though.
def im_u_net(mu1, sigma1, mu2, sigma2, set, H=256, W=256, channels=2, kshape=(3, 3)):
    MOD = set["params"]["MOD"]

    inputs = layers.Input(shape=(H, W, channels))

    conv1 = CompConv2D(24 * MOD)(inputs)
    conv1 = CompConv2D(24 * MOD)(conv1)
    conv1 = CompConv2D(24 * MOD)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = CompConv2D(32 * MOD)(pool1)
    conv2 = CompConv2D(32 * MOD)(conv2)
    conv2 = CompConv2D(32 * MOD)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = CompConv2D(64 * MOD)(pool2)
    conv3 = CompConv2D(64 * MOD)(conv3)
    conv3 = CompConv2D(64 * MOD)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = CompConv2D(128 * MOD)(pool3)
    conv4 = CompConv2D(128 * MOD)(conv4)
    conv4 = CompConv2D(128 * MOD)(conv4)

    up1 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = CompConv2D(64 * MOD)(up1)
    conv5 = CompConv2D(64 * MOD)(conv5)
    conv5 = CompConv2D(64 * MOD)(conv5)

    up2 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = CompConv2D(32 * MOD)(up2)
    conv6 = CompConv2D(32 * MOD)(conv6)
    conv6 = CompConv2D(32 * MOD)(conv6)

    up3 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = CompConv2D(24 * MOD)(up3)
    conv7 = CompConv2D(24 * MOD)(conv7)
    conv7 = CompConv2D(24 * MOD)(conv7)

    # conv8 = layers.Conv2D(2, (1, 1), activation='linear')(conv7)
    conv8 = CompConv2D(1)(conv7)
    res1 = layers.Add()([conv8, inputs])
    res1_scaled = layers.Lambda(lambda res1: (res1 * sigma1 + mu1))(res1)

    rec1 = layers.Lambda(ifft_layer)(res1_scaled)
    final = layers.Lambda(lambda rec1: (rec1 - mu2) / sigma2)(rec1)

    model = Model(inputs=inputs, outputs=final)
    return model

    # U-Net model.


def re_u_net(mu1, sigma1, mu2, sigma2, H=256, W=256, channels=2, kshape=(3, 3)):

    inputs = Input(shape=(H, W, channels))

    conv1 = Conv2D(48, kshape, activation="relu", padding="same")(inputs)
    conv1 = Conv2D(48, kshape, activation="relu", padding="same")(conv1)
    conv1 = Conv2D(48, kshape, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kshape, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, kshape, activation="relu", padding="same")(conv2)
    conv2 = Conv2D(64, kshape, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, kshape, activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, kshape, activation="relu", padding="same")(conv3)
    conv3 = Conv2D(128, kshape, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kshape, activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, kshape, activation="relu", padding="same")(conv4)
    conv4 = Conv2D(256, kshape, activation="relu", padding="same")(conv4)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, kshape, activation="relu", padding="same")(up1)
    conv5 = Conv2D(128, kshape, activation="relu", padding="same")(conv5)
    conv5 = Conv2D(128, kshape, activation="relu", padding="same")(conv5)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, kshape, activation="relu", padding="same")(up2)
    conv6 = Conv2D(64, kshape, activation="relu", padding="same")(conv6)
    conv6 = Conv2D(64, kshape, activation="relu", padding="same")(conv6)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(48, kshape, activation="relu", padding="same")(up3)
    conv7 = Conv2D(48, kshape, activation="relu", padding="same")(conv7)
    conv7 = Conv2D(48, kshape, activation="relu", padding="same")(conv7)

    conv8 = layers.Conv2D(2, (1, 1), activation="linear")(conv7)
    res1 = layers.Add()([conv8, inputs])
    res1_scaled = layers.Lambda(lambda res1: (res1 * sigma1 + mu1))(res1)

    rec1 = layers.Lambda(ifft_layer)(res1_scaled)
    final = layers.Lambda(lambda rec1: (rec1 - mu2) / sigma2)(rec1)

    model = Model(inputs=inputs, outputs=final)
    return model
