from pathlib import Path
from comp_unet.Functions import (
    schedule,
    get_brains,
    mask_gen,
    nrmse,
    im_u_net,
    re_u_net,
)
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from hydra import initialize, compose
import glob

# Test schedule function
def test_schedule():
    assert schedule(1, 2) == 2


# Test laoding settings file
def test_cfgload():
    with initialize(version_base=None, config_path="../Inputs"):
        cfg = compose(config_name="settings")
        assert cfg["params"]["UNIT_CONFIRM"] == 66


# Tests that masks are generated
def test_maskgen():
    with initialize(version_base=None, config_path="../Inputs"):
        cfg = compose(config_name="settings")

    ADDR = Path.cwd()

    mask_gen(ADDR, cfg)

    masks = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["MASKS_ADDR"])))
    assert cfg["params"]["NUM_MASKS"] == len(masks)


# Tests MRI scan loading/processing
def test_loading():
    with initialize(version_base=None, config_path="../Inputs"):
        cfg = compose(config_name="settings")

    ADDR = Path.cwd()

    (
        mask,
        stats,
        kspace_train,
        image_train,
        kspace_val,
        image_val,
        kspace_test,
        image_test,
    ) = get_brains(cfg, ADDR)

    rec_train = np.copy(image_train)
    image_train = image_train[:, :, :, 0]
    image_train = np.expand_dims(image_train, axis=3)
    image_val = image_val[:, :, :, 0]
    image_val = np.expand_dims(image_val, axis=3)
    image_test = image_test[:, :, :, 0]
    image_test = np.expand_dims(image_test, axis=3)

    assert rec_train.shape == (cfg["params"]["NUM_TRAIN"], 256, 256, 2)
    assert image_train.shape == (cfg["params"]["NUM_TRAIN"], 256, 256, 1)
    assert kspace_train.shape == (cfg["params"]["NUM_TRAIN"], 256, 256, 2)
    assert image_val.shape == (cfg["params"]["NUM_VAL"], 256, 256, 1)
    assert kspace_val.shape == (cfg["params"]["NUM_VAL"], 256, 256, 2)
    assert image_test.shape == (cfg["params"]["NUM_TEST"], 256, 256, 1)
    assert kspace_test.shape == (cfg["params"]["NUM_TEST"], 256, 256, 2)


def test_imcreate():
    with initialize(version_base=None, config_path="../Inputs"):
        cfg = compose(config_name="settings")

    ADDR = Path.cwd()

    (
        mask,
        stats,
        kspace_train,
        image_train,
        kspace_val,
        image_val,
        kspace_test,
        image_test,
    ) = get_brains(cfg, ADDR)

    rec_train = np.copy(image_train)
    image_train = image_train[:, :, :, 0]
    image_train = np.expand_dims(image_train, axis=3)
    image_val = image_val[:, :, :, 0]
    image_val = np.expand_dims(image_val, axis=3)
    image_test = image_test[:, :, :, 0]
    image_test = np.expand_dims(image_test, axis=3)

    model = im_u_net(stats[0], stats[1], stats[2], stats[3], cfg)
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(optimizer=opt, loss=nrmse)

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

    model.fit(
        combined,
        epochs=cfg["params"]["EPOCHS"],
        steps_per_epoch=rec_train.shape[0] / cfg["params"]["BATCH_SIZE"],
        verbose=1,
        validation_data=(kspace_val, image_val),
        callbacks=[lrs, mc, es, csvl],
    )

    model.save(ADDR / cfg["addrs"]["IMMODEL_ADDR"])
    model2 = tf.keras.models.load_model(
        ADDR / cfg["addrs"]["IMMODEL_ADDR"],
        custom_objects={"nrmse": nrmse, "CompConv2D": CompConv2D},
    )

    assert model == model2


def test_recreate():
    with initialize(version_base=None, config_path="../Inputs"):
        cfg = compose(config_name="settings")

    ADDR = Path.cwd()

    (
        mask,
        stats,
        kspace_train,
        image_train,
        kspace_val,
        image_val,
        kspace_test,
        image_test,
    ) = get_brains(cfg, ADDR)

    rec_train = np.copy(image_train)
    image_train = image_train[:, :, :, 0]
    image_train = np.expand_dims(image_train, axis=3)
    image_val = image_val[:, :, :, 0]
    image_val = np.expand_dims(image_val, axis=3)
    image_test = image_test[:, :, :, 0]
    image_test = np.expand_dims(image_test, axis=3)

    model = re_u_net(stats[0], stats[1], stats[2], stats[3], cfg)
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(optimizer=opt, loss=nrmse)

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

    model.fit(
        combined,
        epochs=cfg["params"]["EPOCHS"],
        steps_per_epoch=rec_train.shape[0] / cfg["params"]["BATCH_SIZE"],
        verbose=1,
        validation_data=(kspace_val, image_val),
        callbacks=[lrs, mc, es, csvl],
    )

    model.save(ADDR / cfg["addrs"]["REMODEL_ADDR"])
    model2 = tf.keras.models.load_model(
        ADDR / cfg["addrs"]["REMODEL_ADDR"],
        custom_objects={"nrmse": nrmse},
    )

    assert model == model2
