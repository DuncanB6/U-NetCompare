# Basic Unet model that mirrors the complex Unet, but does not use the CompConv2D layer.
# June 21, 2022


# Imports
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

    # Finds root address, will need to be checked in ARC.
    ADDR = Path.cwd()  # /Users/duncan.boyd/Documents/WorkCode/workvenv
    ADDR = ADDR / "UofC2022"

    # Imports functions
    sys.path.append(str(ADDR / set["addrs"]["FUNC_ADDR"]))
    from Functions import get_brains, re_u_net, nrmse, schedule

    logging.info("Initialized re UNet")
    init_time = time.time()

    # Loads data
    logging.info("Loading data")
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
    logging.info("Compiling UNet")
    model = re_u_net(stats[0], stats[1], stats[2], stats[3])
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(optimizer=opt, loss=nrmse)

    # Callbacks to manage training
    lrs = tf.keras.callbacks.LearningRateScheduler(schedule)
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ADDR / set["addrs"]["RECHEC_ADDR"]),
        mode="min",
        monitor="val_loss",
        save_best_only=True,
    )
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
    csvl = tf.keras.callbacks.CSVLogger(
        str(ADDR / set["addrs"]["RECSV_ADDR"]), append=False, separator="|"
    )

    # Fits model using training data, validation data
    logging.info("Fitting UNet")
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
    model.save(ADDR / set["addrs"]["REMODEL_ADDR"])
    model = tf.keras.models.load_model(
        ADDR / set["addrs"]["REMODEL_ADDR"], custom_objects={"nrmse": nrmse}
    )

    # Makes predictions
    logging.info("Evaluating UNet")
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

    logging.info("Done")

    return


# Name guard
if __name__ == "__main__":

    # Runs the main program above
    main()
