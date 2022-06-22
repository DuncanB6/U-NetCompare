# June 22, 2022

# Status:
# This is the beginning of a complex U-Net. Code is written, returns poor quality images.
# Hopefully poor quality is due to training practices limited by local hardware and not by method.

# This branch has hydra implemente for the config file. However,
# Hydra's logging is blocking my own and doesn't seem to be working.
# Otherwise, code functions well.

# To do:
# Find out how to load models with custom layers and functions
# Data augmentation ( from keras.preprocessing.image import ImageDataGenerator) )
# Revise scheduler
# Fix logging, integrate with hydra
# Unit testing, containerization, turning code into package

# Name guard
if __name__ == "__main__":

    # Imports
    import os
    import time
    from datetime import datetime
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import logging
    import sys
    from pathlib import Path
    import hydra
    from omegaconf import DictConfig

    # Import settings with hydra
    @hydra.main(
        version_base=None,
        config_path="../Inputs",
        config_name="settings",
    )
    def main(cfg: DictConfig) -> None:
        set = cfg

        # This line may be necessary for logging
        # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        # Finds root address, will need to be checked in ARC.
        ADDR = Path.cwd()  # /Users/duncan.boyd/Documents/WorkCode/workvenv
        ADDR = ADDR / "UofC2022"

        # Imports functions
        sys.path.append(str(ADDR / set["addrs"]["FUNC_ADDR"]))
        from Functions import get_brains, im_u_net, nrmse, schedule

        """# Initializes logging
        logging.basicConfig(
            filename=str(ADDR / set["addrs"]["IMLOG_ADDR"]),
            filemode="w",
            format="%(name)s - %(levelname)s - %(message)s",
            level=logging.DEBUG,
        )
        logging.getLogger("matplotlib").setLevel(logging.WARNING)"""
        logging.debug("Initialized im UNet")
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

        # Declares, compiles, fits the model.
        logging.debug("Compiling UNet")
        model = im_u_net(stats[0], stats[1], stats[2], stats[3], set)
        opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-7)
        model.compile(optimizer=opt, loss=nrmse)

        # Callbacks to manage training
        lrs = tf.keras.callbacks.LearningRateScheduler(schedule)
        mc = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ADDR / set["addrs"]["IMCHEC_ADDR"]),
            mode="min",
            monitor="val_loss",
            save_best_only=True,
        )
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, mode="min"
        )
        csvl = tf.keras.callbacks.CSVLogger(
            str(ADDR / set["addrs"]["IMCSV_ADDR"]), append=False, separator="|"
        )

        # Fits model using training data, validation data
        logging.debug("Fitting UNet")
        model.fit(
            kspace_train,
            image_train,
            validation_data=(kspace_val, image_val),
            batch_size=set["params"]["BATCH_SIZE"],
            epochs=set["params"]["EPOCHS"],
            callbacks=[lrs, mc, es, csvl],
        )

        # Saves model
        # Note: Loading does not work due to custom layers
        model.save(ADDR / set["addrs"]["IMMODEL_ADDR"])

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

        # Displays predictions (Not necessary for ARC)
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.imshow((255.0 - image_test[0]), cmap="Greys")
        plt.subplot(1, 2, 2)
        plt.imshow((255.0 - predictions[0]), cmap="Greys")
        file_name = "im_" + str(int(end_time - init_time)) + ".jpg"
        # plt.savefig(str(ADDR / 'Outputs' / file_name))
        plt.show()

        logging.debug("Done")

        return

    # Runs the main program above
    main()
