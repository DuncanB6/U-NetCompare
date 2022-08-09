# This is an ARC compatible file which fits and saves two unets, real and complex.

# Imports
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
from unet_compare.real_unet import real_main
from unet_compare.comp_unet import comp_main
from unet_compare.functions import get_brains, mask_gen
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
        dec_train,
        rec_train,
        dec_val,
        rec_val,
    ) = get_brains(cfg, ADDR)

    # Calls both models
    comp_main(
        cfg,
        ADDR,
        mask,
        stats,
        rec_train,
        dec_val,
        rec_val,
    )
    real_main(
        cfg,
        ADDR,
        mask,
        stats,
        rec_train,
        dec_val,
        rec_val,
    )

    return


# Name guard
if __name__ == "__main__":

    # Runs the main program above
    main()
