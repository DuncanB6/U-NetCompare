# June 27, 2022

# Status:
# This is the beginning of a complex U-Net. Returns medium quality images.
# Variable results most likely due to small sample sizes.

# To do:
# Stupid CompUNet keeps returning black images :(
# Find out how to load models with custom layers and functions (error due to modifyable channels out)
# Revise scheduler (unsure what this should be doing)
# Unit testing, containerization, turning code into package (optional, would like to review with mike)
# Determine the actual experiments/training to be done on ARC (once other tasks are complete)

# Imports
from pathlib import Path
import sys
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt

# Import settings with hydra
@hydra.main(
    version_base=None,
    config_path="../Inputs",
    config_name="settings",
)
def main(cfg: DictConfig):
    # Finds root address, will need to be checked in ARC.
    ADDR = Path.cwd()  # /Users/duncan.boyd/Documents/WorkCode/workvenv
    ADDR = ADDR / "UofC2022"
    READDR = ADDR / "UNet"
    IMADDR = ADDR / "CompUNet"

    # Imports both UNets
    sys.path.append(str(READDR))
    sys.path.append(str(IMADDR))
    sys.path.append(str(ADDR / cfg["addrs"]["FUNC_ADDR"]))
    from UNet import remain
    from CompUNet import immain
    from Functions import get_brains, mask_gen

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

    """plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(
        np.abs(np.fft.ifft2(kspace_train[0, :, :, 0] + 1j * kspace_train[0, :, :, 1])),
        cmap="Greys",
    )
    plt.subplot(1, 2, 2)
    plt.imshow(image_train[0], cmap="Greys")
    plt.show()"""

    # Calls both models

    immodel = immain(
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
    remodel = remain(
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
