# Complex UNet that uses a custom layer.

# Imports
import time
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import sys


def immain(
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

    # Imports functions
    sys.path.append(str(ADDR / cfg["addrs"]["FUNC_ADDR"]))
    from Functions import im_u_net, nrmse, schedule, data_aug

    logging.info("Initialized im UNet")
    init_time = time.time()

    # Declares, compiles, fits the model.
    logging.info("Compiling UNet")
    model = im_u_net(stats[0], stats[1], stats[2], stats[3], cfg)
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(optimizer=opt, loss=nrmse)

    # Callbacks to manage training
    lrs = tf.keras.callbacks.LearningRateScheduler(schedule)
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ADDR / cfg["addrs"]["IMCHEC_ADDR"]),
        mode="min",
        monitor="val_loss",
        save_best_only=True,
    )
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
    csvl = tf.keras.callbacks.CSVLogger(
        str(ADDR / cfg["addrs"]["IMCSV_ADDR"]), append=False, separator="|"
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
    # Note: Loading does not work due to custom layers. It want an unpit for out_channels
    # while loading, but this is determined in the UNet.
    # Note: Code below this point will be removed for ARC testing
    model.save(ADDR / cfg["addrs"]["IMMODEL_ADDR"])
    """model = tf.keras.models.load_model(
        ADDR / cfg["addrs"]["IMMODEL_ADDR"],
        custom_objects={"nrmse": nrmse, "CompConv2D": CompConv2D},
    )"""

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
    file_name = "im_" + str(int(end_time - init_time)) + ".jpg"
    # plt.savefig(str(ADDR / 'Outputs' / file_name))
    plt.show()

    logging.info("Done")

    return model
