# Functions that support both UNets. Includes the UNets themselves and the custom layers.

import os
import tensorflow.compat.v1 as tf
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
import numpy as np
import glob
from keras import backend as K
from keras.models import Model
import logging
from keras.preprocessing.image import ImageDataGenerator
import random
import sigpy.mri as sp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.disable_v2_behavior()

# Creates a number of masks (modifiable in settings) with a 22% poisson disk.
def mask_gen(ADDR, cfg):

    files = glob.glob(str(ADDR / cfg["addrs"]["MASKS"]))
    for f in files:
        os.remove(f)

    for k in range(cfg["params"]["NUM_MASKS"]):
        mask = sp.poisson(
            img_shape=(256, 256),
            accel=5.5,
            dtype=int,
            crop_corner=True,
        )

        ones = 0
        zeros = 0

        for i in range(256):
            for j in range(256):
                if mask[i][j] > 0:
                    ones += 1
                else:
                    zeros += 1

        ratio = ones / zeros * 100.0
        ratio = float(f"{ratio:.4f}")

        filename = "/mask" + str(int(k)) + "_" + str(ratio) + ".npy"
        filename = cfg["addrs"]["MASK_SAVE"] + filename
        np.save(
            str(ADDR / filename),
            mask,
        )

    return


# Return an image generator which generates augmented images
def data_aug(rec_train, mask, stats, cfg):
    seed = 905
    image_datagen1 = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.075,
        height_shift_range=0.075,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest",
    )

    image_datagen2 = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.075,
        height_shift_range=0.075,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest",
    )

    image_datagen1.fit(rec_train[:, :, :, 0, np.newaxis], augment=True, seed=seed)
    image_datagen2.fit(rec_train[:, :, :, 1, np.newaxis], augment=True, seed=seed)

    image_gen1 = image_datagen1.flow(
        rec_train[:, :, :, 0, np.newaxis],
        batch_size=cfg["params"]["BATCH_SIZE"],
        seed=seed,
    )
    image_gen2 = image_datagen1.flow(
        rec_train[:, :, :, 1, np.newaxis],
        batch_size=cfg["params"]["BATCH_SIZE"],
        seed=seed,
    )

    def combine_generator(gen1, gen2, mask, stats):
        while True:
            rec_real = gen1.next()
            rec_imag = gen2.next()
            kspace = np.fft.fft2(rec_real[:, :, :, 0] + 1j * rec_imag[:, :, :, 0])
            kspace2 = np.zeros((kspace.shape[0], kspace.shape[1], kspace.shape[2], 2))
            kspace2[:, :, :, 0] = kspace.real
            kspace2[:, :, :, 1] = kspace.imag
            kspace2[
                :, mask[int(random.randint(0, (cfg["params"]["NUM_MASKS"] - 1)))], :
            ] = 0
            kspace2 = (kspace2 - stats[0]) / stats[1]
            rec = rec_real[:, :, :, 0]
            rec = np.expand_dims(rec, axis=3)
            yield (kspace2, rec)

    return combine_generator(image_gen1, image_gen2, mask, stats)


# Scheduler, currently just a stolen one as I don't know what I should be going for.
def schedule(epoch, lr):
    return lr / (1 + epoch)


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
def get_brains(cfg, ADDR):

    # Note: In train, one file is (174, 256, 256).
    kspace_files_train = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["TRAIN"])))
    kspace_files_val = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["VAL"])))
    kspace_files_test = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["TEST"])))

    logging.info("train scans: " + str(len(kspace_files_train)))
    logging.info("val scans: " + str(len(kspace_files_val)))
    logging.info("test scans: " + str(len(kspace_files_test)))
    logging.debug("Scans loaded")

    shape = (256, 256)
    norm = np.sqrt(shape[0] * shape[1])

    mask = np.zeros((cfg["params"]["NUM_MASKS"], shape[0], shape[1]))
    masks = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["MASKS"])))
    for i in range(len(masks)):
        mask[i] = np.load(masks[i])
    mask = mask.astype(int)
    logging.info("masks: " + str(len(mask)))
    # mask = np.load(str(ADDR / cfg["addrs"]["MASK_ADDR"]))

    # Get number of samples
    ntrain = 0
    for ii in range(len(kspace_files_train)):
        ntrain += np.load(kspace_files_train[ii]).shape[0]

    # Load train data
    image_train = np.zeros((ntrain, shape[0], shape[1], 2))
    kspace_train = np.zeros((ntrain, shape[0], shape[1], 2))
    aux_counter = 0
    for ii in range(len(kspace_files_train)):
        aux_kspace = np.load(kspace_files_train[ii]) / norm
        aux = aux_kspace.shape[0]
        aux2 = np.fft.ifft2(aux_kspace[:, :, :, 0] + 1j * aux_kspace[:, :, :, 1])
        image_train[aux_counter : aux_counter + aux, :, :, 0] = aux2.real
        image_train[aux_counter : aux_counter + aux, :, :, 1] = aux2.imag
        kspace_train[aux_counter : aux_counter + aux, :, :, 0] = aux_kspace[:, :, :, 0]
        kspace_train[aux_counter : aux_counter + aux, :, :, 1] = aux_kspace[:, :, :, 1]
        aux_counter += aux

    # Shuffle training
    indexes = np.arange(image_train.shape[0], dtype=int)
    np.random.shuffle(indexes)
    image_train = image_train[indexes]
    kspace_train = kspace_train[indexes]
    kspace_train[:, mask[int(random.randint(0, cfg["params"]["NUM_MASKS"] - 1))], :] = 0

    kspace_train = kspace_train[: cfg["params"]["NUM_TRAIN"], :, :, :]
    image_train = image_train[: cfg["params"]["NUM_TRAIN"], :, :, :]

    logging.info("kspace train: " + str(kspace_train.shape))
    logging.info("image train: " + str(image_train.shape))

    # Get number of samples
    nval = 0
    for ii in range(len(kspace_files_val)):
        nval += np.load(kspace_files_val[ii]).shape[0]

    # Load val data
    image_val = np.zeros((nval, shape[0], shape[1], 2))
    kspace_val = np.zeros((nval, shape[0], shape[1], 2))
    aux_counter = 0
    for ii in range(len(kspace_files_val)):
        aux_kspace = np.load(kspace_files_val[ii]) / norm
        aux = aux_kspace.shape[0]
        aux2 = np.fft.ifft2(aux_kspace[:, :, :, 0] + 1j * aux_kspace[:, :, :, 1])
        image_val[aux_counter : aux_counter + aux, :, :, 0] = aux2.real
        image_val[aux_counter : aux_counter + aux, :, :, 1] = aux2.imag
        kspace_val[aux_counter : aux_counter + aux, :, :, 0] = aux_kspace[:, :, :, 0]
        kspace_val[aux_counter : aux_counter + aux, :, :, 1] = aux_kspace[:, :, :, 1]
        aux_counter += aux

    # Shuffle valing
    indexes = np.arange(image_val.shape[0], dtype=int)
    np.random.shuffle(indexes)
    image_val = image_val[indexes]
    kspace_val = kspace_val[indexes]
    kspace_val[:, mask[int(random.randint(0, cfg["params"]["NUM_MASKS"] - 1))], :] = 0

    kspace_val = kspace_val[: cfg["params"]["NUM_VAL"], :, :, :]
    image_val = image_val[: cfg["params"]["NUM_VAL"], :, :, :]

    logging.info("kspace val: " + str(kspace_val.shape))
    logging.info("image val: " + str(image_val.shape))

    # Get number of samples
    ntest = 0
    for ii in range(len(kspace_files_test)):
        ntest += np.load(kspace_files_test[ii]).shape[0]

    # Load test data
    image_test = np.zeros((ntest, shape[0], shape[1], 2))
    kspace_test = np.zeros((ntest, shape[0], shape[1], 2))
    aux_counter = 0
    for ii in range(len(kspace_files_test)):
        aux_kspace = np.load(kspace_files_test[ii]) / norm
        aux = aux_kspace.shape[0]
        aux2 = np.fft.ifft2(aux_kspace[:, :, :, 0] + 1j * aux_kspace[:, :, :, 1])
        image_test[aux_counter : aux_counter + aux, :, :, 0] = aux2.real
        image_test[aux_counter : aux_counter + aux, :, :, 1] = aux2.imag
        kspace_test[aux_counter : aux_counter + aux, :, :, 0] = aux_kspace[:, :, :, 0]
        kspace_test[aux_counter : aux_counter + aux, :, :, 1] = aux_kspace[:, :, :, 1]
        aux_counter += aux

    # Shuffle testing
    indexes = np.arange(image_test.shape[0], dtype=int)
    np.random.shuffle(indexes)
    image_test = image_test[indexes]
    kspace_test = kspace_test[indexes]
    kspace_test[:, mask[int(random.randint(0, cfg["params"]["NUM_MASKS"] - 1))], :] = 0

    kspace_test = kspace_test[: cfg["params"]["NUM_TEST"], :, :, :]
    image_test = image_test[: cfg["params"]["NUM_TEST"], :, :, :]

    logging.info("kspace test: " + str(kspace_test.shape))
    logging.info("image test: " + str(image_test.shape))

    logging.debug("Scans formatted")

    # Save k-space and image domain stats
    stats = np.zeros(4)
    stats[0] = kspace_train.mean()
    stats[1] = kspace_train.std()
    aux = np.abs(image_train[:, :, :, 0] + 1j * image_train[:, :, :, 1])
    stats[2] = aux.mean()
    stats[3] = aux.std()
    np.save(str(ADDR / cfg["addrs"]["STATS"]), stats)

    return (
        mask,
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
        self.out_channels = out_channels
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
        config = {
            "convreal": self.convreal,
            "convimag": self.convimag,
            "out_channels": self.out_channels,
        }
        base_config = super(CompConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# U-Net model. Includes kspace domain U-Net and IFFT.
# Note: Filters are halved to maintain structure, but they probably shouldn't be. Although this may reduce performance, I think it's probably fine for testing.
# A variable (MOD) was added to make testing easier. At small scale, MOD = 1 worked best.
# Question: Can a purely kspace model be trained, with no reference to image domain? For this code, I've assumed no (also just for convenience viewing results).
# I'm pretty sure it can though.
def comp_unet_model(
    mu1, sigma1, mu2, sigma2, cfg, H=256, W=256, channels=2, kshape=(3, 3)
):
    MOD = cfg["params"]["MOD"]

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

    # conv8 = layers.Conv2D(2, (1, 1), activation="linear")(conv7)
    conv8 = CompConv2D(1)(conv7)
    res1 = layers.Add()([conv8, inputs])
    res1_scaled = layers.Lambda(lambda res1: (res1 * sigma1 + mu1))(res1)

    rec1 = layers.Lambda(ifft_layer)(res1_scaled)
    final = layers.Lambda(lambda rec1: (rec1 - mu2) / sigma2)(rec1)

    model = Model(inputs=inputs, outputs=final)
    return model


# U-Net model.
def real_unet_model(mu1, sigma1, mu2, sigma2, H=256, W=256, channels=2, kshape=(3, 3)):

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
