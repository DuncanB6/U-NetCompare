Duncan Boyd, July 4, 2022

Description: This repository is for all code written as a summer student in the Seaman MRI Center 2022. 
Comp UNet is the code I'm working on, and is as suggested a complex UNet, as well as a regular UNet.
The other UNet is a regular UNet.

---

Status: Currently working to improve usabiity and ready it for large scale testing. Once that's done, I'll be working 
towards sending larger scale jobs to ARC.

---

ARC instructions:

export PATH=~/software/miniconda3/bin:$PATH

source activate testvenv

sbatch arc_script.slurm

---

Reminder for Duncan: 

git add . 

git commit -m "_message_" 

git push origin main 

git pull origin main

pip3 list – format=freeze > requirements.txt


