
# June 17, 2022

# Status: This is the beginning of a complex U-Net. Code is written, returns poor quality images.
# Hopefully poor quality is due to training practices limited by local hardware and not by method.
# Data processing is limited, but does use validation. Considering data augmentation.
# Will need significant updates to deploy at larger scale.

# To do:
# Determine and implement evalutation methods
# Log data while training
# Find a way to put import and global vars in another file
# + Mike's other stuff

if __name__ == '__main__':
    
    # Is there a good way to only import this block in one file (instead of having this block of code at the top
    # of every file), or as a function?
    # Like a header, but for python. I haven't found a good way to do it yet, but I'm sure it's doable.
    import os
    import time
    from re import I
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow import keras
    from keras import layers
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    from keras import backend as K
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras.preprocessing.image import ImageDataGenerator
    import random
    import logging
    import yaml
    import sys
    from pathlib import *

    ROOT_DIR = Path.cwd() # /Users/duncan.boyd/Documents/WorkCode/workvenv

    # Imports global vars from settings YAML file.
    # Same deal as above, is it possible to only see this block of code in one file?
    with open("/Users/duncan.boyd/Documents/WorkCode/workvenv/UofC2022/config/settings.yaml", "r") as yamlfile:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    '''EPOCHS = data['EPOCHS']
    MOD = data['MOD']
    BATCH_SIZE = data['BATCH_SIZE']
    NUM_TRAIN = data['NUM_TRAIN']
    NUM_VAL = data['NUM_VAL']
    NUM_TEST = data['NUM_TEST']
    ADDR = data['ADDR']'''

    # Imports my functions
    sys.path.append('/Users/duncan.boyd/Documents/WorkCode/workvenv/UofC2022/Functions')
    from Functions import get_brains, im_u_net, nrmse

    # Initializes logging
    logging.basicConfig(filename='/Users/duncan.boyd/Documents/WorkCode/workvenv/UofC2022/Data/CompUNet.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.debug('Initialized im UNet')
    init_time = time.time()

    # Loads data
    logging.debug('Loading data')
    stats, kspace_train, image_train, kspace_val, image_val, kspace_test, image_test = get_brains(NUM_TRAIN, NUM_VAL, NUM_TEST, ADDR)

    # Declare, compile, fit the model.
    logging.debug('Compiling UNet')
    model = im_u_net(stats[0],stats[1],stats[2],stats[3])
    opt = tf.keras.optimizers.Adam(lr=1e-3,decay = 1e-7)
    model.compile(optimizer=opt, loss=nrmse)

    # Some tools are skipped here (model loading, early stopping) as they aren't effective/necessary for small scale testing. 

    # Fits model using training data, validation data.
    logging.debug('Fitting UNet')
    model.fit(kspace_train, image_train, validation_data=(kspace_val, image_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

    # Makes predictions
    logging.debug('Evaluating UNet')
    predictions = model.predict(kspace_test)
    print(predictions.shape)

    end_time = time.time()
    logging.info("total time: " + str(int(end_time-init_time)))

    # Displays predictions
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.imshow((255.0 - image_test[0]), cmap='Greys')
    plt.subplot(1,2,2)
    plt.imshow((255.0 - predictions[0]), cmap='Greys')
    # plt.savefig("/Users/duncan.boyd/Documents/WorkCode/workvenv/UofC2022/SmallScaleTest/im_"+str(EPOCHS)+"_"+str(NUM_TRAIN)+"_"+str(MOD)+"_"+str(int(end_time-init_time))+".jpg")
    plt.show()

    logging.debug("Done")


