# Basic Unet model that mirrors the complex Unet, but does not use the CompConv2D layer.
# June 20, 2022

if __name__ == "__main__":

    # Is there a good way to only import this block in one file (instead of having this block of code at the top
    # of every file), or as a function?
    # Like a header, but for python. I haven't found a good way to do it yet, but I'm sure it's doable.
    import os
    import time
    from datetime import datetime
    import tensorflow.compat.v1 as tf
    import keras as ks
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
    with open(ADDR / "Data/settings.yaml", "r") as yamlfile:
        set = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Imports my functions
    sys.path.append(str(ADDR / "Functions"))
    from Functions import get_brains, re_u_net, nrmse

    # Initializes logging
    logging.basicConfig(
        filename=str(ADDR / "Data/UNet.log"),
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
    ) = get_brains(set, ADDR)

    # Declare, compile, fit the model.
    logging.debug("Compiling UNet")
    model = re_u_net(stats[0], stats[1], stats[2], stats[3])
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(optimizer=opt, loss=nrmse)

    # Some tools are skipped here (model loading, early stopping) as they aren't effective/necessary for small scale testing.

    # Fits model using training data, validation data.
    logging.debug("Fitting UNet")
    model.fit(
        kspace_train,
        image_train,
        validation_data=(kspace_val, image_val),
        batch_size=set["params"]["BATCH_SIZE"],
        epochs=set["params"]["EPOCHS"],
    )

    # Save model
    model.save(ADDR / set["addrs"]["REMODEL_ADDR"])

    # Makes predictions
    logging.debug("Evaluating UNet")
    predictions = model.predict(kspace_test)
    print(predictions.shape)

    # Provides endtime logging info
    end_time = time.time()
    now = datetime.now()
    time_finished = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("total time: " + str(int(end_time - init_time)))
    logging.info("time completed: " + time_finished)

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
