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
from unet_compare.functions import nrmse, CompConv2D, get_test, metrics

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

    # Following 3 blocks convert kspace to image domain.
    # When implemented in a function, this code returned grey images.
    aux = np.fft.ifft2(kspace_test[:, :, :, 0] + 1j * kspace_test[:, :, :, 1])
    image = np.copy(kspace_test)
    image[:, :, :, 0] = aux.real
    image[:, :, :, 1] = aux.imag
    kspace_test = image

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
    real_pred = real_model.predict(kspace_test)
    print(real_pred.shape)

    print(kspace_test[0, 0, 0, 0].type())
    print(real_pred[0, 0, 0, 0].type())

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

    # High score is better for SSIM, PSNR
    # Low is better for NRMSE
    # Should be ballpark (0.978 +/- 0.076, 1.827 +/- 1.112, 35.543 +/- 3.239)
    '''metrics(image_test, kspace_test)
    metrics(image_test, comp_pred)
    metrics(image_test, real_pred)'''


# Name guard
if __name__ == "__main__":

    # Runs the main program above
    main()
