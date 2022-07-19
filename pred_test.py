# This program is used to evaluate existing models. At the moment, it
# only shows the models predictions, but in the future will be used
# do do more detailed analysis.

# Requires two existing unets, displays prediction images.

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
from unet_compare.functions import nrmse, CompConv2D, get_test

# Collects test data and uses it to evaluate existing models.
@hydra.main(
    version_base=None,
    config_path="../UofC2022/inputs/configs",
    config_name="settings_1",
)
def main(cfg: DictConfig):

    ADDR = Path.cwd()

    # Loads data
    (
        kspace_test,
        image_test,
    ) = get_test(cfg, ADDR)

    image_test = image_test[:, :, :, 0]
    image_test = np.expand_dims(image_test, axis=3)

    comp_model = tf.keras.models.load_model(
        ADDR / cfg["addrs"]["COMP_ARC"],
        custom_objects={"nrmse": nrmse, "CompConv2D": CompConv2D},
    )

    real_model = tf.keras.models.load_model(
        ADDR / cfg["addrs"]["REAL_ARC"], custom_objects={"nrmse": nrmse}
    )

    # Makes predictions
    comp_pred = comp_model.predict(kspace_test)
    print(comp_pred.shape)

    # Makes predictions
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


# Name guard
if __name__ == "__main__":

    # Runs the main program above
    main()
