
# Duncan Boyd, 11\05\2022
# Uses a U-Net to reconstruct undersampled (kind of) photographs taken from a large dataset.

# Libraries
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras as ks
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, concatenate, UpSampling2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as iio
import glob
import cv2

# Constants
BRAINS = 1
PICS = 170
SAMP_RATE = 5
EPOCHS = 5
TOTAL_PICS = 50
TEST_PICS = 10

# U-Net model.
def u_net(H=256,W=256,channels = 1,kshape = (3,3)):
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
    
    outputs = Conv2D(1, (1, 1), activation='linear')(conv7)

    final = Model(inputs=inputs, outputs=outputs)
    return final

# Displays messages to confirm initialization.
print("\nInitializing...")
print("Using Keras version", ks.__version__, "\n")

# Loads MRI data to the brains array.
imshape = (256,256)
norm = np.sqrt(imshape[0]*imshape[1])
brains = []
non_brains = []
for brain in glob.glob('/Users/duncan.boyd/Documents/WorkCode/workvenv/MRIPractice/Train/*.npy') :
    kspace = np.load(brain)/norm
    rec = np.abs(np.fft.ifft2(kspace[:,:,:,0]+1j*kspace[:,:,:,1])).astype(np.float64)
    if rec.shape == (170, 256, 256) :
        brains.append(rec)
brains = np.array(brains)
print(brains.shape)

# Creates two normalized sets for training and testing.
brains = brains / 35.46896
brains = np.expand_dims(brains, axis=-1)
train_set_rec = brains[0, 60:(60+TOTAL_PICS), :, :]
train_set_dec = train_set_rec.copy()
test_set_rec = brains[1, 80:(80+TEST_PICS), :, :]
test_set_dec = test_set_rec.copy()

# Undersamples the deconstructed (initial) datasets.
for i in range(TOTAL_PICS) :
    for j in range(256) :
        for k in range(256):
            if j % SAMP_RATE == 0 or k % SAMP_RATE == 0 :
                train_set_dec[i, j, k] = 0
for i in range(TEST_PICS) :
    for j in range(256) :
        for k in range(256):
            if j % SAMP_RATE == 0 or k % SAMP_RATE == 0 :
                test_set_dec[i, j, k] = 0

# Prints shapes of
print("Data in in range 0<x<1")
print("Complete training: ", train_set_rec.shape)
print("Incomplete training: ", train_set_dec.shape)
print("Test set rec: ", test_set_rec.shape)
print("Test set dec: ", test_set_dec.shape)

# Calls U-Net model, compiles. 
u_net =u_net(H=256,W=256,channels = 1,kshape = (3,3))
u_net.compile(optimizer='adam',loss='mse',metrics=[ks.metrics.RootMeanSquaredError()])

# Fits model using training data.
u_net.fit(train_set_dec, train_set_rec, epochs=EPOCHS)

# Predicts a set of reconstructed images.
predictions = u_net.predict(test_set_dec)
print(predictions.shape)

# Plots an individual image from test and predictions.
fig_num = 0
while(fig_num != -1) :
    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(test_set_dec[fig_num], cmap='Greys')
    plt.subplot(1,3,2)
    plt.imshow(test_set_rec[fig_num], cmap='Greys')
    plt.subplot(1,3,3)
    plt.imshow(predictions[fig_num], cmap='Greys')
    plt.show()
    fig_num = int(input('Figure number:'))

exit()