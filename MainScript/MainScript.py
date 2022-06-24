# Imports
from pathlib import Path
import sys
import hydra
from omegaconf import DictConfig

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
    from UNet import remain
    from CompUNet import immain

    immain(cfg, ADDR)
    remain(cfg, ADDR)
    return


# Name guard
if __name__ == "__main__":

    # Runs the main program above
    main()
