# This program is used to evaluate existing models.

# Requires two existing unets, displays prediction recs.

# scp -r duncan.boyd@arc.ucalgary.ca:home/metrics_test metrics_test
# scp -r unet_compare duncan.boyd@arc.ucalgary.ca:home

# Imports
import numpy as np
import tensorflow as tf
import glob
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from unet_compare.functions import nrmse, CompConv2D, get_test, metrics, mask_gen

# Collects test data and uses it to evaluate existing models.
@hydra.main(
    version_base=None,
    config_path="../UofC2022/inputs/configs",
    config_name="test_settings",
)
def main(cfg: DictConfig):

    ADDR = Path.cwd()

    mask_gen(ADDR, cfg)

    # Loads data
    (
        dec_test,
        rec_test,
    ) = get_test(cfg, ADDR)

    plt.imshow((255.0 - dec_test[0, :, :, 0]), cmap='Greys')
    plt.show()
    plt.imshow((255.0 - rec_test[0, :, :, 0]), cmap='Greys')
    plt.show()

    comp_models = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["COMP_ARC"])))
    real_models = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["REAL_ARC"])))

    real_ssim_sum = 0.0
    real_nrmse_sum = 0.0
    real_psnr_sum = 0.0
    comp_ssim_sum = 0.0
    comp_nrmse_sum = 0.0
    comp_psnr_sum = 0.0

    metrics_file = open(ADDR / cfg['addrs']['METRICS'], 'w')
    metrics_file.write("Metrics:\n")
    metrics_file.close()

    for i in range(len(comp_models)):

        comp_model = tf.keras.models.load_model(
            ADDR / comp_models[i],
            custom_objects={"nrmse": nrmse, "CompConv2D": CompConv2D},
        )

        real_model = tf.keras.models.load_model(
            ADDR / real_models[i], custom_objects={"nrmse": nrmse}
        )

        # Makes predictions
        comp_pred = comp_model.predict(dec_test)
        real_pred = real_model.predict(dec_test)

        # Displays predictions (Not necessary for large scale)
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 4, 1)
        plt.imshow((255.0 - rec_test[0, :, :, 0]), cmap="Greys")
        plt.axis("off")
        plt.subplot(1, 4, 2)
        plt.imshow((255.0 - comp_pred[0, :, :, 0]), cmap="Greys")
        plt.axis("off")
        plt.subplot(1, 4, 3)
        plt.imshow((255.0 - real_pred[0, :, :, 0]), cmap="Greys")
        plt.axis("off")
        plt.subplot(1, 4, 4)
        plt.imshow((255.0 - dec_test[0, :, :, 0]), cmap="Greys")
        plt.axis("off")
        plt.show()

        comp_pred = comp_pred / np.max(np.abs(comp_pred[:, :, :, 0] + 1j * comp_pred[:, :, :, 1]))
        real_pred = real_pred / np.max(np.abs(real_pred[:, :, :, 0] + 1j * real_pred[:, :, :, 1]))

        # High score is better for SSIM, PSNR
        # Low is better for NRMSE
        # Should be ballpark (0.978 +/- 0.076, 1.827 +/- 1.112, 35.543 +/- 3.239)
        metrics_file = open(ADDR / cfg['addrs']['METRICS'], 'a')
        metric = metrics(rec_test, comp_pred)
        metrics_file.write("\n\nComplex: ")
        metrics_file.write(str(comp_models[i]))
        metrics_file.write("\nSSIM: %.3f +/- %.3f" %(metric[:,0].mean(), metric[:,0].std()))
        metrics_file.write("\nNRMSE: %.3f +/- %.3f" %(metric[:,1].mean(),metric[:,1].std()))
        metrics_file.write("\nPSNR: %.3f +/- %.3f" %(metric[:,2].mean(), metric[:,2].std()))
        comp_ssim_sum += metric[:, 0].mean()
        comp_nrmse_sum += metric[:, 1].mean()
        comp_psnr_sum += metric[:, 2].mean()

        metric = metrics(rec_test, real_pred)
        metrics_file.write("\n\nReal: ")
        metrics_file.write(str(real_models[i]))
        metrics_file.write("\nSSIM: %.3f +/- %.3f" %(metric[:,0].mean(), metric[:,0].std()))
        metrics_file.write("\nNRMSE: %.3f +/- %.3f" %(metric[:,1].mean(),metric[:,1].std()))
        metrics_file.write("\nPSNR: %.3f +/- %.3f" %(metric[:,2].mean(), metric[:,2].std()))
        real_ssim_sum += metric[:, 0].mean()
        real_nrmse_sum += metric[:, 1].mean()
        real_psnr_sum += metric[:, 2].mean()
        
        metrics_file.close()

    metrics_file = open(ADDR / cfg['addrs']['METRICS'], 'a')
    metrics_file.write("\n\nComp avgs: ")
    metrics_file.write("\nSSIM: %.3f NRMSE: %.3f pSNR: %.3f" %(comp_ssim_sum / len(comp_models), comp_nrmse_sum / len(comp_models), comp_psnr_sum / len(comp_models)))
    metrics_file.write("\n\nReal avgs: ")
    metrics_file.write("\nSSIM: %.3f NRMSE: %.3f pSNR: %.3f" %(real_ssim_sum / len(real_models), real_nrmse_sum / len(real_models), real_psnr_sum / len(real_models)))
    metrics_file.close()
    return

# Name guard
if __name__ == "__main__":

    # Runs the main program above
    main()
