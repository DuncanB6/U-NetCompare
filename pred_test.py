# This program is used to evaluate existing models.
# Note: It can also be modified to work with ARC, but may not be necessary.
# To modify, mostly just change the hydra header then debug.

# High score is better for SSIM, PSNR
# Low is better for NRMSE
# Should be ballpark (0.978 +/- 0.076, 1.827 +/- 1.112, 35.543 +/- 3.239) (from w-net, so the u-net will be worst)

# Inputs: 
# test dataset
# inputs/configs/test_settings yaml file
# At least one trained pair of models including models and training logs in metrics_test

# Outputs: 
# System log in outputs
# metrics text file in metrics test

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

# Import settings with hydra
@hydra.main(
    version_base=None,
    config_path="../UofC2022/inputs/configs",
    config_name="test_settings",
)
def main(cfg: DictConfig):

    # Finds working directory address, used with cfg addresses
    ADDR = Path.cwd()

    # Creates masks
    mask_gen(ADDR, cfg)

    # Loads data
    (
        dec_test,
        rec_test,
    ) = get_test(cfg, ADDR)

    # Plots a sample of training data, fully sampled and undersampled
    plt.imshow((255.0 - dec_test[0, :, :, 0]), cmap='Greys')
    plt.show()
    plt.imshow((255.0 - rec_test[0, :, :, 0]), cmap='Greys')
    plt.show()

    # Gets models and logs for analysis
    comp_models = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["COMP_ARC"])))
    real_models = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["REAL_ARC"])))
    comp_logs = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["COMP_LOG"])))
    real_logs = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["REAL_LOG"])))

    # Initialized stats arrays
    real_ssim_sum = []
    real_nrmse_sum = []
    real_psnr_sum = []
    comp_ssim_sum = []
    comp_nrmse_sum = []
    comp_psnr_sum = []
    comp_conv = []
    real_conv = []

    # Mostly just here to make sure metrics file is clean
    metrics_file = open(ADDR / cfg['addrs']['METRICS'], 'w')
    metrics_file.write("Metrics:\n")
    metrics_file.close()

    # Cycles through pairs of models
    for i in range(len(comp_models)):

        # Loads comp and real models
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

        # Displays predictions
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

        # Normalizes predictions
        comp_pred = comp_pred / np.max(np.abs(comp_pred[:, :, :, 0] + 1j * comp_pred[:, :, :, 1]))
        real_pred = real_pred / np.max(np.abs(real_pred[:, :, :, 0] + 1j * real_pred[:, :, :, 1]))

        # Reads log files to array
        comp_log = open(ADDR / comp_logs[i], 'r')
        real_log = open(ADDR / real_logs[i], 'r')
        comp_losses = comp_log.readlines()
        real_losses = real_log.readlines()

        # Determines the point where val_loss reached a minimum
        comp_min = 2.0
        comp_conv_epoch = 0.0
        for k in range(len(comp_losses))[1:]:
            comp_losses[k] = comp_losses[k].strip('\n')
            comp_vals = comp_losses[k].split('|')
            if float(comp_vals[2]) < comp_min:
                comp_min = float(comp_vals[2])
                comp_conv_epoch = float(comp_vals[0])   
        real_min = 2.0
        real_conv_epoch = 0.0
        for k in range(len(real_losses))[1:]:
            real_losses[k] = real_losses[k].strip('\n')
            real_vals = real_losses[k].split('|')
            if float(real_vals[2]) < real_min:
                real_min = float(real_vals[2])
                real_conv_epoch = float(real_vals[0])

        # Closes logs
        comp_log.close
        real_log.close

        # Gets metrics and writes to metrics file
        # The following mess should probably be a function
        metrics_file = open(ADDR / cfg['addrs']['METRICS'], 'a')
        metric = metrics(rec_test, comp_pred)
        metrics_file.write("\n\nComplex: ")
        metrics_file.write(str(comp_models[i]))
        metrics_file.write("\nSSIM: %.3f +/- %.3f" %(metric[:,0].mean(), metric[:,0].std()))
        metrics_file.write("\nNRMSE: %.3f +/- %.3f" %(metric[:,1].mean(),metric[:,1].std()))
        metrics_file.write("\nPSNR: %.3f +/- %.3f" %(metric[:,2].mean(), metric[:,2].std()))
        metrics_file.write("\nEpochs: %.3f" %(comp_conv_epoch))
        comp_ssim_sum.append(metric[:, 0])
        comp_nrmse_sum.append(metric[:, 1])
        comp_psnr_sum.append(metric[:, 2])
        comp_conv.append(comp_conv_epoch)

        metric = metrics(rec_test, real_pred)
        metrics_file.write("\n\nReal: ")
        metrics_file.write(str(real_models[i]))
        metrics_file.write("\nSSIM: %.3f +/- %.3f" %(metric[:,0].mean(), metric[:,0].std()))
        metrics_file.write("\nNRMSE: %.3f +/- %.3f" %(metric[:,1].mean(),metric[:,1].std()))
        metrics_file.write("\nPSNR: %.3f +/- %.3f" %(metric[:,2].mean(), metric[:,2].std()))
        metrics_file.write("\nEpochs: %.3f" %(real_conv_epoch))
        real_ssim_sum.append(metric[:, 0])
        real_nrmse_sum.append(metric[:, 1])
        real_psnr_sum.append(metric[:, 2])
        real_conv.append(real_conv_epoch)
        
        metrics_file.close()

    # Changes lists to np arrays
    comp_ssim_sum = np.asarray(comp_ssim_sum)
    comp_psnr_sum = np.asarray(comp_psnr_sum)
    comp_nrmse_sum = np.asarray(comp_nrmse_sum)
    comp_conv = np.asarray(comp_conv)

    real_ssim_sum = np.asarray(real_ssim_sum)
    real_psnr_sum = np.asarray(real_psnr_sum)
    real_nrmse_sum = np.asarray(real_nrmse_sum)
    real_conv = np.asarray(real_conv)

    # Writes averages to metrics file
    metrics_file = open(ADDR / cfg['addrs']['METRICS'], 'a')
    metrics_file.write("\n\nComp avgs: ")
    metrics_file.write("\nSSIM: %.3f +/- %.3f" %(comp_ssim_sum.mean(), comp_ssim_sum.std()))
    metrics_file.write("\nNRMSE: %.3f +/- %.3f" %(comp_nrmse_sum.mean(),comp_nrmse_sum.std()))
    metrics_file.write("\nPSNR: %.3f +/- %.3f" %(comp_psnr_sum.mean(), comp_psnr_sum.std()))
    metrics_file.write("\nEpochs: %.3f +/- %.3f" %(comp_conv.mean(), comp_conv.std()))
    metrics_file.write("\n\nReal avgs: ")
    metrics_file.write("\nSSIM: %.3f +/- %.3f" %(real_ssim_sum.mean(), real_ssim_sum.std()))
    metrics_file.write("\nNRMSE: %.3f +/- %.3f" %(real_nrmse_sum.mean(),real_nrmse_sum.std()))
    metrics_file.write("\nPSNR: %.3f +/- %.3f" %(real_psnr_sum.mean(), real_psnr_sum.std()))
    metrics_file.write("\nEpochs: %.3f +/- %.3f" %(real_conv.mean(), real_conv.std()))
    metrics_file.close()
    return

# Name guard
if __name__ == "__main__":

    # Runs the main program above
    main()
