from pathlib import Path
from unet_compare.functions import (
    schedule,
    get_brains,
    mask_gen,
    nrmse,
    comp_unet_model,
    real_unet_model,
)
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from hydra import initialize, compose
import glob

# Test schedule function
def test_schedule():
    assert schedule(1, 2) == 2


# Test laoding settings file
def test_cfgload():
    with initialize(version_base=None, config_path="../Inputs"):
        cfg = compose(config_name="settings")
        assert cfg["params"]["UNIT_CONFIRM"] == 66


# Tests that masks are generated
def test_maskgen():
    with initialize(version_base=None, config_path="../Inputs"):
        cfg = compose(config_name="settings")

    ADDR = Path.cwd()

    mask_gen(ADDR, cfg)

    masks = np.asarray(glob.glob(str(ADDR / cfg["addrs"]["MASKS"])))
    assert cfg["params"]["NUM_MASKS"] == len(masks)
