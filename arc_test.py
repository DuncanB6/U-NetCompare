# This is an ARC compatible file which fits two unets, real and complex.

# Imports
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
from unet_compare.real_unet import real_main
from unet_compare.comp_unet import comp_main
from unet_compare.functions import get_brains, mask_gen

# Import settings with hydra
@hydra.main(
    version_base=None,
    config_path="/home/duncan.boyd/home/inputs",
)
def main(cfg: DictConfig):

    cfg = cfg["configs"]  # Add to run direct from command line with various configs

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
        rec_train,
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
        rec_train,
    )
    return


# Name guard
if __name__ == "__main__":

    # Runs the main program above
    main()
