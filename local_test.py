# Aug 5, 2022

# This program fits two unets and displays results. Used for local testing.

# Imports
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from unet_compare.real_unet import real_main
from unet_compare.comp_unet import comp_main
from unet_compare.functions import get_brains, mask_gen, normalize
import logging

# Import settings with hydra
@hydra.main(
    version_base=None,
    config_path="../UofC2022/inputs/configs",
    config_name="settings_1",
)
def main(cfg: DictConfig):

    logging.info("Settings version: " + str(cfg["params"]["UNIT_CONFIRM"]))

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

    # Calls both models
    comp_model = comp_main(
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
    comp_model.summary()
    real_model = real_main(
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
    real_model.summary()

    # Makes predictions
    logging.info("Evaluating UNet")
    comp_pred = comp_model.predict(kspace_test)

    # Makes predictions
    logging.info("Evaluating UNet")
    real_pred = real_model.predict(kspace_test)

    # Displays predictions (Not necessary for ARC)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 4, 1)
    plt.imshow((255.0 - image_test[0, :, :, 0]), cmap="Greys")
    plt.subplot(1, 4, 2)
    plt.imshow((255.0 - comp_pred[0, :, :, 0]), cmap="Greys")
    plt.subplot(1, 4, 3)
    plt.imshow((255.0 - real_pred[0, :, :, 0]), cmap="Greys")
    plt.subplot(1, 4, 4)
    plt.imshow((255.0 - kspace_test[0, :, :, 0]), cmap="Greys")
    plt.show()

    return


# Name guard
if __name__ == "__main__":

    # Runs the main program above
    main()
