# This is an ARC compatible file which fits two unets, real and complex.

# Imports
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
from unet_compare.real_unet import real_main
from unet_compare.comp_unet import comp_main
from unet_compare.functions import get_brains, mask_gen
import numpy as np
import tensorflow as tf

# Import settings with hydra
@hydra.main(
    version_base=None,
    config_path="/home/duncan.boyd/home/inputs",
)
def main(cfg: DictConfig):

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    cfg = cfg["configs"]  # Add to run direct from command line with various configs

    logging.info("Settings version: " + str(cfg["params"]["UNIT_CONFIRM"]))

    # Finds root address, will need to be checked in ARC.
    ADDR = Path.cwd()  # /Users/duncan.boyd/Documents/WorkCode/workvenv/UofC2022

    # Creates masks
    mask_gen(ADDR, cfg)

    # Loads data
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

    # Following 3 blocks convert kspace to image domain.
    # When implemented in a function, this code returned grey images.
    aux = np.fft.ifft2(kspace_train[:, :, :, 0] + 1j * kspace_train[:, :, :, 1])
    image = np.copy(kspace_train)
    image[:, :, :, 0] = aux.real
    image[:, :, :, 1] = aux.imag
    kspace_train = image

    aux = np.fft.ifft2(kspace_test[:, :, :, 0] + 1j * kspace_test[:, :, :, 1])
    image = np.copy(kspace_test)
    image[:, :, :, 0] = aux.real
    image[:, :, :, 1] = aux.imag
    kspace_test = image

    aux = np.fft.ifft2(kspace_val[:, :, :, 0] + 1j * kspace_val[:, :, :, 1])
    image = np.copy(kspace_val)
    image[:, :, :, 0] = aux.real
    image[:, :, :, 1] = aux.imag
    kspace_val = image

    # Calls both models
    immodel = comp_main(
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
    )
    remodel = real_main(
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
    )

    return


# Name guard
if __name__ == "__main__":

    # Runs the main program above
    main()
