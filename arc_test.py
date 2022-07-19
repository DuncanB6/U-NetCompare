# This is an ARC compatible file which fits two unets, real and complex.

# Imports
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
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
