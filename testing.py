

# Imports
import numpy as np
import tensorflow as tf
import glob
import logging
import random
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from unet_compare.functions import (
    nrmse,
    CompConv2D,
)

def get_test(cfg, ADDR):

    kspace_files_test = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["TEST"])))

    logging.info("test scans: " + str(len(kspace_files_test)))
    logging.debug("Scans loaded")

    shape = (256, 256)
    norm = np.sqrt(shape[0] * shape[1])

    mask = np.zeros((cfg["params"]["NUM_MASKS"], shape[0], shape[1]))
    masks = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["MASKS"])))
    for i in range(len(masks)):
        mask[i] = np.load(masks[i])
    mask = mask.astype(int)
    logging.info("masks: " + str(len(mask)))

    # Get number of samples
    ntest = 0
    for ii in range(len(kspace_files_test)):
        ntest += np.load(kspace_files_test[ii]).shape[0]

    # Load test data
    image_test = np.zeros((ntest, shape[0], shape[1], 2))
    kspace_test = np.zeros((ntest, shape[0], shape[1], 2))
    aux_counter = 0
    for ii in range(len(kspace_files_test)):
        aux_kspace = np.load(kspace_files_test[ii]) / norm
        aux = aux_kspace.shape[0]
        aux2 = np.fft.ifft2(aux_kspace[:, :, :, 0] + 1j * aux_kspace[:, :, :, 1])
        image_test[aux_counter : aux_counter + aux, :, :, 0] = aux2.real
        image_test[aux_counter : aux_counter + aux, :, :, 1] = aux2.imag
        kspace_test[aux_counter : aux_counter + aux, :, :, 0] = aux_kspace[:, :, :, 0]
        kspace_test[aux_counter : aux_counter + aux, :, :, 1] = aux_kspace[:, :, :, 1]
        aux_counter += aux

    # Shuffle testing
    indexes = np.arange(image_test.shape[0], dtype=int)
    np.random.shuffle(indexes)
    image_test = image_test[indexes]
    kspace_test = kspace_test[indexes]
    kspace_test[
        :, mask[int(random.randint(0, (cfg["params"]["NUM_MASKS"] - 1)))], :
    ] = 0

    kspace_test = kspace_test[: cfg["params"]["NUM_TEST"], :, :, :]
    image_test = image_test[: cfg["params"]["NUM_TEST"], :, :, :]

    logging.info("kspace test: " + str(kspace_test.shape))
    logging.info("image test: " + str(image_test.shape))

    logging.debug("Scans formatted")

    return (
        kspace_test,
        image_test,
    )

@hydra.main(
    version_base=None,
    config_path="../UofC2022/inputs/configs",
    config_name="settings_2",
)
def main(cfg: DictConfig):

    ADDR = Path.cwd()

    # Loads data
    (
        kspace_test,
        image_test,
    ) = get_test(cfg, ADDR)

    comp_model = tf.keras.models.load_model(
        ADDR / cfg["addrs"]["COMP_MODEL"],
        custom_objects={"nrmse": nrmse, "CompConv2D": CompConv2D},
    )

    real_model = tf.keras.models.load_model(
        ADDR / cfg["addrs"]["REAL_MODEL"], custom_objects={"nrmse": nrmse}
    )

    # Makes predictions
    logging.info("Evaluating UNet")
    comp_pred = comp_model.predict(kspace_test)
    print(comp_pred.shape)

    # Makes predictions
    logging.info("Evaluating UNet")
    real_pred = comp_model.predict(kspace_test)
    print(real_pred.shape)

    # Displays predictions (Not necessary for ARC)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 4, 1)
    plt.imshow((255.0 - image_test[0]), cmap="Greys")
    plt.subplot(1, 4, 2)
    plt.imshow((255.0 - comp_pred[0]), cmap="Greys")
    plt.subplot(1, 4, 3)
    plt.imshow((255.0 - real_pred[0]), cmap="Greys")
    plt.subplot(1, 4, 4)
    plt.imshow(
        (
            255.0
            - np.abs(
                np.fft.ifft2(kspace_test[0, :, :, 0] + 1j * kspace_test[0, :, :, 1])
            )
        ),
        cmap="Greys",
    )
    plt.show()


