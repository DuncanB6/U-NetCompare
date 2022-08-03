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
from unet_compare.functions import nrmse, CompConv2D, get_test, metrics, normalize

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

    # Following lines of code are for debugging regarding data normalization
    print(image_test.max(), image_test.min())

    a_bounds = (image_test.min(), image_test.max())
    cd_bounds = (comp_pred.min(), comp_pred.max())
    rd_bounds = (real_pred.min(), real_pred.max())
    comp_pred = comp_pred.astype(np.float64)
    real_pred = real_pred.astype(np.float64)

    comp_pred = normalize(comp_pred, cd_bounds, a_bounds)
    real_pred = normalize(real_pred, rd_bounds, a_bounds)
    print(comp_pred.max(), comp_pred.min())
    print(real_pred.max(), real_pred.min())

    # comp_pred = np.log(1+np.abs(comp_pred + 1j*comp_pred))
    # real_pred = np.log(1+np.abs(real_pred + 1j*real_pred))

    print(image_test[0, 0, 80:90, :])
    print(comp_pred[0, 0, 80:90, :])

    print(np.mean(comp_pred))
    print(np.mean(image_test))
    print(np.mean(kspace_test))

    # High score is better for SSIM, PSNR
    # Low is better for NRMSE
    # Should be ballpark (0.978 +/- 0.076, 1.827 +/- 1.112, 35.543 +/- 3.239)
    metrics(image_test, kspace_test)
    metrics(image_test, comp_pred)
    metrics(image_test, real_pred)


# Name guard
if __name__ == "__main__":

    # Runs the main program above
    main()
