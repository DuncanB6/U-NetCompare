# This is an ARC compatible file which fits and saves two unets, real and complex.

# Inputs: 
# train, val datasets
# inputs/configs/settings_* yaml file (specified in SLURM file)

# Outputs: 
# Models, checkpoints and logs in output file
# System log in outputs

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

    # Puts cfg library in the right place and finds your operating directory
    cfg = cfg["configs"]
    ADDR = Path.cwd()

    # Initial logging
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    logging.info("Settings version: " + str(cfg["params"]["UNIT_CONFIRM"]))

    # Creates masks
    mask_gen(ADDR, cfg)

    # Loads train, val data
    # Note: dec_train is not used because of image augmentation
    (
        mask,
        stats,
        dec_train,
        rec_train,
        dec_val,
        rec_val,
    ) = get_brains(cfg, ADDR)

    # Calls both models to train
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
