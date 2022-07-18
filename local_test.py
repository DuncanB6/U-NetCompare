# July 4, 2022

# This program fits two unets and displays results. Used for local testing.

# Status:
# This is the beginning of a complex U-Net. Returns medium quality images.
# Variable results most likely due to small sample sizes.

# To do:
# Expand on unit tests (optional)
# Determine the actual experiments/training to be done on ARC (once other tasks are complete)
# Run ARC tests

# Imports
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from unet_compare.real_unet import real_main
from unet_compare.comp_unet import comp_main
from unet_compare.functions import get_brains, mask_gen
import logging

# Import settings with hydra

# For debugging:
# config_path="../UofC2022/inputs/configs",
# config_name="settings_1",

# For multiple configs from command "python3 main_script.py +configs=settings_1":
# NOTE: Make sure to uncomment first line in main.
# config_path="../UofC2022/inputs",


@hydra.main(
    version_base=None,
    config_path="../UofC2022/inputs/configs",
    config_name="settings_1",
)
def main(cfg: DictConfig):

    # cfg = cfg["configs"]  # Add to run direct from command line with various configs

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

    # Block that reverts arrays to the way my code processes them.
    rec_train = np.copy(image_train)
    image_train = image_train[:, :, :, 0]
    image_train = np.expand_dims(image_train, axis=3)
    image_val = image_val[:, :, :, 0]
    image_val = np.expand_dims(image_val, axis=3)
    image_test = image_test[:, :, :, 0]
    image_test = np.expand_dims(image_test, axis=3)

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
        rec_train,
    )
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
        rec_train,
    )
    real_model.summary()
    comp_model.summary()

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

    return


# Name guard
if __name__ == "__main__":

    # Runs the main program above
    main()
