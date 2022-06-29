# Basic Unet model that mirrors the complex Unet, but does not use the CompConv2D layer.

# Imports
import time
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from comp_unet.Functions import re_u_net, nrmse, schedule, data_aug


def remain(
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

    logging.info("Initialized re UNet")
    init_time = time.time()

    # Declares, compiles, fits the model.
    logging.info("Compiling UNet")
    model = re_u_net(stats[0], stats[1], stats[2], stats[3])
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(optimizer=opt, loss=nrmse)

    # Callbacks to manage training
    lrs = tf.keras.callbacks.LearningRateScheduler(schedule)
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ADDR / cfg["addrs"]["RECHEC_ADDR"]),
        mode="min",
        monitor="val_loss",
        save_best_only=True,
    )
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
    csvl = tf.keras.callbacks.CSVLogger(
        str(ADDR / cfg["addrs"]["RECSV_ADDR"]), append=False, separator="|"
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
        callbacks=[lrs, mc, es, csvl],
    )
    model.summary()

    # Saves model
    # Note: Loading does not work due to custom layers
    # Note: Code below this point will be removed for ARC testing
    model.save(ADDR / cfg["addrs"]["REMODEL_ADDR"])
    model = tf.keras.models.load_model(
        ADDR / cfg["addrs"]["REMODEL_ADDR"], custom_objects={"nrmse": nrmse}
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
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow((255.0 - image_test[0]), cmap="Greys")
    plt.subplot(1, 2, 2)
    plt.imshow((255.0 - predictions[0]), cmap="Greys")
    file_name = "re_" + str(int(end_time - init_time)) + ".jpg"
    plt.savefig(str(ADDR / "Outputs" / file_name))
    plt.show()

    logging.info("Done")

    return model