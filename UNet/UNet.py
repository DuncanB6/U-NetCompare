
# Basic Unet model that mirrors the complex Unet, but does not use the CompConv2D layer.
# June 14, 2022

# Imports
import os
import time
from re import I
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras
from keras import layers
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, concatenate, UpSampling2D
import numpy as np
import matplotlib.pyplot as plt
import glob
from keras import backend as K
from keras.models import Model
import random

# Variables
EPOCHS = 1
MOD = 1
BATCH_SIZE = 1
num_train = 10 # max 4254
num_val = 5 # max 1700
num_test = 5 # max 1700
addr = '/Users/duncan.boyd/Documents/WorkCode/workvenv/MRIPractice/'

# Loss function
def nrmse(y_true, y_pred):
    denom = K.sqrt(K.mean(K.square(y_true), axis=(1,2,3)))
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=(1,2,3)))\
    /denom

# IFFT layer, used in u_net.
def ifft_layer(kspace):
    real = layers.Lambda(lambda kspace : kspace[:,:,:,0])(kspace)
    imag = layers.Lambda(lambda kspace : kspace[:,:,:,1])(kspace)
    kspace_complex = tf.complex(real,imag)
    rec1 = tf.abs(tf.ifft2d(kspace_complex))
    rec1 = tf.expand_dims(rec1, -1)
    return rec1

# Upgraded version, returns fewer arrays but with a faster and more efficient method.
# Still need to figure out the imaginary image domain stuff. 
def get_brains(num_train, num_val, num_test, addr):

    # Note: In train, one file is (174, 256, 256).
    kspace_files_train = np.asarray(glob.glob(addr+"Train/*.npy")) 
    kspace_files_val = np.asarray(glob.glob(addr+"Val/*.npy"))
    kspace_files_test = np.asarray(glob.glob(addr+"Test/*.npy"))

    print("train scans:", len(kspace_files_train))
    print("val scans:", len(kspace_files_val))
    print("test scans:", len(kspace_files_test))

    samp_mask = np.load(addr+"mask.npy")
    shape = (256, 256)
    norm = np.sqrt(shape[0]*shape[1])

    image_train = []
    kspace_train = []
    count = 0
    for i in random.sample(range(0, len(kspace_files_train)), len(kspace_files_train)):
        brain = np.load(kspace_files_train[i])/norm
        if count >= num_train:
                break
        for i in random.sample(range(0, brain.shape[0]), brain.shape[0]):
            image_train.append(np.abs(np.fft.ifft2(brain[i,:,:,0]+1j*brain[i,:,:,1])).astype(np.float64))
            brain[i, samp_mask, : ] = 0
            kspace_train.append(brain[i].astype(np.float64))
            count += 1
            if count >= num_train:
                break
    kspace_train = np.asarray(kspace_train)
    print("kspace train:", kspace_train.shape)
    image_train = np.asarray(image_train)
    image_train = np.expand_dims(image_train, axis=3)
    print("image train:", image_train.shape)

    image_val = []
    kspace_val = []
    count = 0
    for i in random.sample(range(0, len(kspace_files_val)), len(kspace_files_val)):
        brain = np.load(kspace_files_val[i])/norm
        if count >= num_val:
                break
        for i in random.sample(range(0, brain.shape[0]), brain.shape[0]):
            image_val.append(np.abs(np.fft.ifft2(brain[i,:,:,0]+1j*brain[i,:,:,1])).astype(np.float64))
            brain[i, samp_mask, : ] = 0
            kspace_val.append(brain[i].astype(np.float64))
            count += 1
            if count >= num_val:
                break
    kspace_val = np.asarray(kspace_val)
    print("kspace val:", kspace_val.shape)
    image_val = np.asarray(image_val)
    image_val = np.expand_dims(image_val, axis=3)
    print("image val:", image_val.shape)

    image_test = []
    kspace_test = []
    count = 0
    for i in random.sample(range(0, len(kspace_files_test)), len(kspace_files_test)):
        brain = np.load(kspace_files_test[i])/norm
        if count >= num_test:
                break
        for i in random.sample(range(0, brain.shape[0]), brain.shape[0]):
            image_test.append(np.abs(np.fft.ifft2(brain[i,:,:,0]+1j*brain[i,:,:,1])).astype(np.float64))
            brain[i, samp_mask, : ] = 0
            kspace_test.append(brain[i].astype(np.float64))
            count += 1
            if count >= num_test:
                break
    kspace_test = np.asarray(kspace_test)
    print("kspace test:", kspace_test.shape)
    image_test = np.asarray(image_test)
    image_test = np.expand_dims(image_test, axis=3)
    print("image test:", image_test.shape)

    # Question: image_train seems to be the reconstructed images here, but it's imaginary. Why is image domain imaginary?
    # Especially since the WNet returns a real image. Does the imaginary part in image domain yield any valuable information?
    # For the time being, I'm just going to load the stats manually.
    stats = np.load('/Users/duncan.boyd/Documents/WorkCode/workvenv/WNetPractice/stats.npy')
    '''# save k-space and image domain stats
    stats = np.zeros(4)
    stats[0] = kspace_train.mean()
    stats[1] = kspace_train.std()
    aux = np.abs(image_train[:,:,:,0] +1j*image_train[:,:,:,1])
    stats[2] = aux.mean()
    stats[3] = aux.std()
    np.save(STATS_ADDR, stats)'''

    return stats, kspace_train, image_train, kspace_val, image_val, kspace_test, image_test

# U-Net model.
def u_net(mu1,sigma1,mu2,sigma2, H=256,W=256,channels = 2,kshape = (3,3)):
    inputs = Input(shape=(H,W,channels))

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3],axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2],axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1],axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)

    conv8 = layers.Conv2D(2, (1, 1), activation='linear')(conv7)
    res1 = layers.Add()([conv8,inputs])
    res1_scaled = layers.Lambda(lambda res1 : (res1*sigma1+mu1))(res1)
    
    rec1 = layers.Lambda(ifft_layer)(res1_scaled)
    final = layers.Lambda(lambda rec1 : (rec1-mu2)/sigma2)(rec1)

    model = Model(inputs=inputs, outputs=final)
    return model

if __name__ == '__main__':

    init_time = time.time()
    print("\nInitialized\n")

    stats, kspace_train, image_train, kspace_val, image_val, kspace_test, image_test = get_brains(num_train, num_val, num_test, addr)

    # Declare, compile, fit the model.
    model = u_net(stats[0],stats[1],stats[2],stats[3])
    opt = tf.keras.optimizers.Adam(lr=1e-3,decay = 1e-7)
    model.compile(optimizer=opt, loss=nrmse)

    # Some tools are skipped here (model loading, early stopping) as they aren't effective/necessary for small scale testing. 

    # Fits model using training data, validation data.
    model.fit(kspace_train, image_train, validation_data=(kspace_val, image_val), batch_size=BATCH_SIZE, epochs=EPOCHS)
    model.summary()

    # Makes predictions
    predictions = model.predict(kspace_test)
    print(predictions.shape)

    end_time = time.time()

    # Displays predictions
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.imshow((255.0 - image_test[0]), cmap='Greys')
    plt.subplot(1,2,2)
    plt.imshow((255.0 - predictions[0]), cmap='Greys')
    plt.savefig("/Users/duncan.boyd/Documents/WorkCode/workvenv/UofC2022/SmallScaleTest/re_"+str(EPOCHS)+"_"+str(num_train)+"_"+str(MOD)+"_"+str(int(end_time-init_time))+".jpg")
    plt.show()







