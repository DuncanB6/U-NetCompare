# Complex UNet that uses a custom layer.

# Imports
import time
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import numpy as np
from unet_compare.functions import (
    comp_unet_model,
    nrmse,
    data_aug,
    CompConv2D,
)


def comp_main(
    cfg,
    ADDR,
    mask,
    stats,
    kspace_train,
    image_train,
    kspace_val,
    image_val,
    kspace_test,
    image_test,
    rec_train,
):

    logging.info("Initialized complex UNet with ")
    init_time = time.time()

    # Declares, compiles, fits the model.
    logging.info("Compiling UNet")
    model = comp_unet_model(stats[0], stats[1], stats[2], stats[3], cfg)
    opt = tf.keras.optimizers.Adam(
        lr=cfg["params"]["LR"],
        beta_1=cfg["params"]["BETA_1"],
        beta_2=cfg["params"]["BETA_2"],
    )
    model.compile(optimizer=opt, loss=nrmse)

    # Callbacks to manage training
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ADDR / cfg["addrs"]["COMP_CHEC"]),
        mode="min",
        monitor="val_loss",
        save_best_only=True,
    )
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
    csvl = tf.keras.callbacks.CSVLogger(
        str(ADDR / cfg["addrs"]["COMP_CSV"]), append=False, separator="|"
    )
    combined = data_aug(rec_train, mask, stats, cfg)

    # Fits model using training data, validation data
    logging.info("Fitting UNet")
    model.fit(
        combined,
        epochs=cfg["params"]["EPOCHS"],
        steps_per_epoch=rec_train.shape[0] / cfg["params"]["BATCH_SIZE"],
        verbose=1,
        validation_data=(kspace_val, image_val),
        callbacks=[mc, es, csvl],
    )

    # Saves model
    # Note: Loading does not work due to custom layers. It want an unpit for out_channels
    # while loading, but this is determined in the UNet.
    model.save(ADDR / cfg["addrs"]["COMP_MODEL"])

    # Provides endtime logging info
    end_time = time.time()
    now = datetime.now()
    time_finished = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("total time: " + str(int(end_time - init_time)))
    logging.info("time completed: " + time_finished)
    print("Time:", str(int(end_time - init_time)))

    logging.info("Done")

    return model
