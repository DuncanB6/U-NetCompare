# Basic Unet model that mirrors the complex Unet, but does not use the CompConv2D layer.
# June 20, 2022

if __name__ == "__main__":

    # Is there a good way to only import this block in one file (instead of having this block of code at the top
    # of every file), or as a function?
    # Like a header, but for python. I haven't found a good way to do it yet, but I'm sure it's doable.
    import os
    import time
    import tensorflow.compat.v1 as tf
    import matplotlib.pyplot as plt
    import logging
    import yaml
    import sys
    from pathlib import Path

    # from keras.callbacks import ModelCheckpoint, EarlyStopping
    # from keras.preprocessing.image import ImageDataGenerator

    tf.disable_v2_behavior()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    ADDR = Path.cwd()  # /Users/duncan.boyd/Documents/WorkCode/workvenv
    ADDR = ADDR / "UofC2022"

    # Imports global vars from settings YAML file.
    # Same deal as above, is it possible to only see this block of code in one file?
    with open(ADDR / "Data/settings.yaml", "r") as yamlfile:
        set = yaml.load(yamlfile, Loader=yaml.FullLoader)
    EPOCHS = set["params"]["EPOCHS"]
    MOD = set["params"]["MOD"]
    BATCH_SIZE = set["params"]["BATCH_SIZE"]
    NUM_TRAIN = set["params"]["NUM_TRAIN"]
    NUM_VAL = set["params"]["NUM_VAL"]
    NUM_TEST = set["params"]["NUM_TEST"]

    # Imports my functions
    sys.path.append(str(ADDR / "Functions"))
    from Functions import get_brains, re_u_net, nrmse

    # Initializes logging
    logging.basicConfig(
        filename=str(ADDR / "Data/CompUNet.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.debug("Initialized re UNet")
    init_time = time.time()

    # Loads data
    logging.debug("Loading data")
    (
        stats,
        kspace_train,
        image_train,
        kspace_val,
        image_val,
        kspace_test,
        image_test,
    ) = get_brains(NUM_TRAIN, NUM_VAL, NUM_TEST, ADDR)

    # Declare, compile, fit the model.
    logging.debug("Compiling UNet")
    model = re_u_net(stats[0], stats[1], stats[2], stats[3], MOD)
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(optimizer=opt, loss=nrmse)

    # Some tools are skipped here (model loading, early stopping) as they aren't effective/necessary for small scale testing.

    # Fits model using training data, validation data.
    logging.debug("Fitting UNet")
    model.fit(
        kspace_train,
        image_train,
        validation_data=(kspace_val, image_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    # Makes predictions
    logging.debug("Evaluating UNet")
    predictions = model.predict(kspace_test)
    print(predictions.shape)

    end_time = time.time()
    logging.info("total time: " + str(int(end_time - init_time)))

    # Displays predictions
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.imshow((255.0 - image_test[0]), cmap="Greys")
    plt.subplot(1, 2, 2)
    plt.imshow((255.0 - predictions[0]), cmap="Greys")
    file_name = "re_" + str(int(end_time - init_time)) + ".jpg"
    # plt.savefig(str(ADDR / 'Outputs' / file_name))
    plt.show()

    logging.debug("Done")
