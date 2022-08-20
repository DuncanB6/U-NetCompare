# Functions that support both UNets. Includes the UNets themselves and the custom layers.
# This is the majority of code for this project.
# Because of this, commenting is only done at function top. Good luck.

# Note that the get_test and get_brains functions load all data, even if they only return a part.
# This is a flaw that could be fixed.

import os
from re import L
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
import numpy as np
import glob
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import sigpy.mri as sp
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as norm_root_mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Compares two data sets, prints and returns ssim, psnr, nrmse.
def metrics(ref, pred):

    for ii in range(pred.shape[0]):  
        metrics[ii,0] = ssim(ref[ii].ravel(), pred[ii].ravel(), win_size = ref[ii].size-1)
        metrics[ii,1] = norm_root_mse(ref[ii], pred[ii])
        metrics[ii,2] = psnr(ref[ii], pred[ii], data_range=(ref[ii].max()-ref[ii].min())) 

    metrics[:,1] = metrics[:,1]*100
    print("Metrics:")
    print("SSIM: %.3f +/- %.3f" %(metrics[:,0].mean(), metrics[:,0].std()))
    print("NRMSE: %.3f +/- %.3f" %(metrics[:,1].mean(),metrics[:,1].std()))
    print("PSNR: %.3f +/- %.3f" %(metrics[:,2].mean(), metrics[:,2].std()))

    return metrics

# Gets test data only.
def get_test(cfg, ADDR):

    dec_files_test = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["TEST"])))

    logging.info("test scans: " + str(len(dec_files_test)))
    logging.debug("Scans loaded")

    shape = (256, 256)
    norm = np.sqrt(shape[0] * shape[1])

    mask = np.zeros((cfg["params"]["NUM_MASKS"], shape[0], shape[1]))
    masks = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["MASKS"])))
    for i in range(len(masks)):
        mask[i] = np.load(masks[i])
    mask = mask.astype(bool)
    logging.info("masks: " + str(len(mask)))

    ntest = 0
    for ii in range(len(dec_files_test)):
        ntest += np.load(dec_files_test[ii]).shape[0]

    rec_test = np.zeros((ntest, shape[0], shape[1], 2))
    dec_test = np.zeros((ntest, shape[0], shape[1], 2))
    aux_counter = 0
    for ii in range(len(dec_files_test)):
        dec1 = np.load(dec_files_test[ii]) / norm
        rec1 = np.copy(dec1)
        dec1[:, mask[int(random.randint(0, cfg["params"]["NUM_MASKS"] - 1))], :] = 0
        dec2 = np.fft.ifft2(dec1[:, :, :, 0] + 1j * dec1[:, :, :, 1])
        rec2 = np.fft.ifft2(rec1[:, :, :, 0] + 1j * rec1[:, :, :, 1])
        aux = rec1.shape[0]
        dec_test[aux_counter : aux_counter + aux, :, :, 0] = dec2.real
        dec_test[aux_counter : aux_counter + aux, :, :, 1] = dec2.imag
        rec_test[aux_counter : aux_counter + aux, :, :, 0] = rec2.real
        rec_test[aux_counter : aux_counter + aux, :, :, 1] = rec2.imag
        aux_counter += aux
    
    indexes = np.arange(rec_test.shape[0], dtype=int)
    np.random.shuffle(indexes)
    rec_test = rec_test[indexes]
    dec_test = dec_test[indexes]

    dec_test = dec_test[: cfg["params"]["NUM_TEST"], :, :, :]
    rec_test = rec_test[: cfg["params"]["NUM_TEST"], :, :, :]

    dec_test = dec_test / np.max(np.abs(dec_test[:, :, :, 0] + 1j * dec_test[:, :, :, 1]))
    rec_test = rec_test / np.max(np.abs(rec_test[:, :, :, 0] + 1j * rec_test[:, :, :, 1]))

    dec_test = dec_test.astype('float32')
    rec_test = rec_test.astype('float32')

    logging.info("dec test: " + str(dec_test.shape))
    logging.info("rec test: " + str(rec_test.shape))

    logging.debug("Scans formatted")

    return (
        dec_test,
        rec_test,
    )

# Created a boolean circle mask
def create_circular_mask(h=256, w=256, center=None, radius=16):

    if center is None: # use the middle of the rec
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and rec walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    mask = ~np.fft.fftshift(mask, axes=(0, 1))
    mask = mask.astype(np.bool)
    
    return mask

# Creates a number of masks with a poisson disk and a circular mask
def mask_gen(ADDR, cfg):

    files = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["MASKS"])))
    for f in files:
        os.remove(f)

    for k in range(cfg["params"]["NUM_MASKS"]):
        mask = sp.poisson(
            img_shape=(256, 256),
            accel=cfg["params"]["ACCEL"],
            dtype=int,
            crop_corner=False,
        )

        mask = ~np.fft.fftshift(mask, axes=(0, 1))

        mask = mask + 2
        mask = mask.astype(np.bool)
        mask = mask & create_circular_mask()

        sampling = (1.0*mask.sum()/mask.size) * 100

        filename = "/mask" + str(int(k)) + "_" + str(cfg["params"]["ACCEL"]) + "_" + str(int(sampling)) + ".npy"
        filename = cfg["addrs"]["MASK_SAVE"] + filename
        np.save(
            str(ADDR / filename),
            mask,
        )

    return


# Returns an image generator which generates images, undersampled and complete
def data_aug(rec_train, mask, stats, cfg):
    seed = 905
    rec_datagen1 = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.075,
        height_shift_range=0.075,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest",
    )

    rec_datagen2 = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.075,
        height_shift_range=0.075,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest",
    )

    rec_datagen1.fit(rec_train[:, :, :, 0, np.newaxis], augment=True, seed=seed)
    rec_datagen2.fit(rec_train[:, :, :, 1, np.newaxis], augment=True, seed=seed)

    rec_gen1 = rec_datagen1.flow(
        rec_train[:, :, :, 0, np.newaxis],
        batch_size=cfg["params"]["BATCH_SIZE"],
        seed=seed,
    )
    rec_gen2 = rec_datagen1.flow(
        rec_train[:, :, :, 1, np.newaxis],
        batch_size=cfg["params"]["BATCH_SIZE"],
        seed=seed,
    )

    def combine_generator(gen1, gen2, mask, stats):
        while True:
            rec_real = gen1.next()
            rec_imag = gen2.next()
            rec = np.zeros((rec_real.shape[0], rec_real.shape[1], rec_real.shape[2], 2))
            rec[:, :, :, 0] = rec_real[:, :, :, 0]
            rec[:, :, :, 1] = rec_imag[:, :, :, 0]

            dec = np.fft.fft2(rec_real[:, :, :, 0] + 1j * rec_imag[:, :, :, 0])
            dec2 = np.zeros((dec.shape[0], dec.shape[1], dec.shape[2], 2))
            dec2[:, :, :, 0] = dec.real
            dec2[:, :, :, 1] = dec.imag
            dec2[
                :, mask[int(random.randint(0, (cfg["params"]["NUM_MASKS"] - 1)))], :
            ] = 0

            aux = np.fft.ifft2(dec2[:, :, :, 0] + 1j * dec2[:, :, :, 1])
            dec = np.copy(dec2)
            dec[:, :, :, 0] = aux.real
            dec[:, :, :, 1] = aux.imag

            dec = dec.astype('float32')
            rec = rec.astype('float32')
            
            yield (dec, rec)

    return combine_generator(rec_gen1, rec_gen2, mask, stats)


# Loss function
def nrmse(y_true, y_pred):
    denom = K.sqrt(K.mean(K.square(y_true), axis=(1, 2, 3)))
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=(1, 2, 3))) / denom

# IFFT layer, used in u_net
def ifft_layer(dec):
    real = layers.Lambda(lambda dec: dec[:, :, :, 0])(dec)
    imag = layers.Lambda(lambda dec: dec[:, :, :, 1])(dec)
    dec_complex = tf.complex(real, imag)
    rec1 = tf.abs(tf.ifft2d(dec_complex))
    rec1 = tf.expand_dims(rec1, -1)
    return rec1


# Gets training data and val data
# Note: In train, one file is (174 x 256 x 256). This code is fine with that
def get_brains(cfg, ADDR):

    dec_files_train = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["TRAIN"])))
    dec_files_val = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["VAL"])))

    logging.info("train scans: " + str(len(dec_files_train)))
    logging.info("val scans: " + str(len(dec_files_val)))
    logging.debug("Scans loaded")

    shape = (256, 256)
    norm = np.sqrt(shape[0] * shape[1])

    mask = np.zeros((cfg["params"]["NUM_MASKS"], shape[0], shape[1]))
    masks = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["MASKS"])))
    for i in range(len(masks)):
        mask[i] = np.load(masks[i])
    mask = mask.astype(bool)
    logging.info("masks: " + str(len(mask)))

    ntrain = 0
    for ii in range(len(dec_files_train)):
        ntrain += np.load(dec_files_train[ii]).shape[0]

    rec_train = np.zeros((ntrain, shape[0], shape[1], 2))
    dec_train = np.zeros((ntrain, shape[0], shape[1], 2))
    aux_counter = 0
    for ii in range(len(dec_files_train)):
        dec1 = np.load(dec_files_train[ii]) / norm
        rec1 = np.copy(dec1)
        dec1[:, mask[int(random.randint(0, cfg["params"]["NUM_MASKS"] - 1))], :] = 0
        dec2 = np.fft.ifft2(dec1[:, :, :, 0] + 1j * dec1[:, :, :, 1])
        rec2 = np.fft.ifft2(rec1[:, :, :, 0] + 1j * rec1[:, :, :, 1])
        aux = rec1.shape[0]
        dec_train[aux_counter : aux_counter + aux, :, :, 0] = dec2.real
        dec_train[aux_counter : aux_counter + aux, :, :, 1] = dec2.imag
        rec_train[aux_counter : aux_counter + aux, :, :, 0] = rec2.real
        rec_train[aux_counter : aux_counter + aux, :, :, 1] = rec2.imag
        aux_counter += aux

    indexes = np.arange(rec_train.shape[0], dtype=int)
    np.random.shuffle(indexes)
    rec_train = rec_train[indexes]
    dec_train = dec_train[indexes]

    dec_train = dec_train[: cfg["params"]["NUM_TRAIN"], :, :, :]
    rec_train = rec_train[: cfg["params"]["NUM_TRAIN"], :, :, :]

    dec_train = dec_train / np.max(np.abs(dec_train[:, :, :, 0] + 1j * dec_train[:, :, :, 1]))
    rec_train = rec_train / np.max(np.abs(rec_train[:, :, :, 0] + 1j * rec_train[:, :, :, 1]))

    dec_train = dec_train.astype('float32')
    rec_train = rec_train.astype('float32')

    logging.info("dec train: " + str(dec_train.shape))
    logging.info("rec train: " + str(rec_train.shape))

    nval = 0
    for ii in range(len(dec_files_val)):
        nval += np.load(dec_files_val[ii]).shape[0]

    rec_val = np.zeros((nval, shape[0], shape[1], 2))
    dec_val = np.zeros((nval, shape[0], shape[1], 2))
    aux_counter = 0
    for ii in range(len(dec_files_val)):
        dec1 = np.load(dec_files_val[ii]) / norm
        rec1 = np.copy(dec1)
        dec1[:, mask[int(random.randint(0, cfg["params"]["NUM_MASKS"] - 1))], :] = 0
        dec2 = np.fft.ifft2(dec1[:, :, :, 0] + 1j * dec1[:, :, :, 1])
        rec2 = np.fft.ifft2(rec1[:, :, :, 0] + 1j * rec1[:, :, :, 1])
        aux = rec1.shape[0]
        dec_val[aux_counter : aux_counter + aux, :, :, 0] = dec2.real
        dec_val[aux_counter : aux_counter + aux, :, :, 1] = dec2.imag
        rec_val[aux_counter : aux_counter + aux, :, :, 0] = rec2.real
        rec_val[aux_counter : aux_counter + aux, :, :, 1] = rec2.imag
        aux_counter += aux

    indexes = np.arange(rec_val.shape[0], dtype=int)
    np.random.shuffle(indexes)
    rec_val = rec_val[indexes]
    dec_val = dec_val[indexes]
    dec_val[:, mask[int(random.randint(0, cfg["params"]["NUM_MASKS"] - 1))], :] = 0

    dec_val = dec_val[: cfg["params"]["NUM_VAL"], :, :, :]
    rec_val = rec_val[: cfg["params"]["NUM_VAL"], :, :, :]

    dec_val = dec_val / np.max(np.abs(dec_val[:, :, :, 0] + 1j * dec_val[:, :, :, 1]))
    rec_val = rec_val / np.max(np.abs(rec_val[:, :, :, 0] + 1j * rec_val[:, :, :, 1]))

    dec_val = dec_val.astype('float32')
    rec_val = rec_val.astype('float32')

    logging.info("dec val: " + str(dec_val.shape))
    logging.info("rec val: " + str(rec_val.shape))

    logging.debug("Scans formatted")

    stats = np.zeros(4)
    stats[0] = dec_train.mean()
    stats[1] = dec_train.std()
    aux = np.abs(rec_train[:, :, :, 0] + 1j * rec_train[:, :, :, 1])
    stats[2] = aux.mean()
    stats[3] = aux.std()
    np.save(str(ADDR / cfg["addrs"]["STATS"]), stats)

    return (
        mask,
        stats,
        dec_train,
        rec_train,
        dec_val,
        rec_val,
    )


# Custom complex convolution.
# Uses algebra below. I've used "|" to denote a two channel array, and "f" to denote a variable that is a part of a filter.
# (R | I) * (Rf | If) = Or | Oi = (R * Rf - I * If) | (I * Rf + R * If)
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


# U-Net model. Uses custom complex layer.
def comp_unet_model(
    cfg, H=256, W=256, channels=2, kshape=(3, 3)
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

    conv8 = layers.Conv2D(2, (1, 1), activation="linear")(conv7)

    model = Model(inputs=inputs, outputs=conv8)
    return model


# U-Net model.
def real_unet_model(
    cfg, H=256, W=256, channels=2, kshape=(3, 3)
):
    RE_MOD = cfg["params"]["RE_MOD"]

    inputs = Input(shape=(H, W, channels))

    conv1 = Conv2D(48 * RE_MOD, kshape, activation="relu", padding="same")(inputs)
    conv1 = Conv2D(48 * RE_MOD, kshape, activation="relu", padding="same")(conv1)
    conv1 = Conv2D(48 * RE_MOD, kshape, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64 * RE_MOD, kshape, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64 * RE_MOD, kshape, activation="relu", padding="same")(conv2)
    conv2 = Conv2D(64 * RE_MOD, kshape, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128 * RE_MOD, kshape, activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128 * RE_MOD, kshape, activation="relu", padding="same")(conv3)
    conv3 = Conv2D(128 * RE_MOD, kshape, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256 * RE_MOD, kshape, activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256 * RE_MOD, kshape, activation="relu", padding="same")(conv4)
    conv4 = Conv2D(256 * RE_MOD, kshape, activation="relu", padding="same")(conv4)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128 * RE_MOD, kshape, activation="relu", padding="same")(up1)
    conv5 = Conv2D(128 * RE_MOD, kshape, activation="relu", padding="same")(conv5)
    conv5 = Conv2D(128 * RE_MOD, kshape, activation="relu", padding="same")(conv5)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64 * RE_MOD, kshape, activation="relu", padding="same")(up2)
    conv6 = Conv2D(64 * RE_MOD, kshape, activation="relu", padding="same")(conv6)
    conv6 = Conv2D(64 * RE_MOD, kshape, activation="relu", padding="same")(conv6)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(48 * RE_MOD, kshape, activation="relu", padding="same")(up3)
    conv7 = Conv2D(48 * RE_MOD, kshape, activation="relu", padding="same")(conv7)
    conv7 = Conv2D(48 * RE_MOD, kshape, activation="relu", padding="same")(conv7)

    conv8 = layers.Conv2D(2, (1, 1), activation="linear")(conv7)

    model = Model(inputs=inputs, outputs=conv8)
    return model
