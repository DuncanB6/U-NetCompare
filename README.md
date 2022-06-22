Duncan Boyd, June 14, 2022

Description: This repository is for all code written as a summer student in the Seaman MRI Center 2022. 
Comp UNet is the code I'm working on, and is as suggested a complex UNet.
The other UNet is a regular UNet.

---

Status: Currently working to improve usabiity and ready it for large scale testing. Once that's done, I'll be working 
towards sending larger scale jobs to ARC. I've registered for an account, but have been unable to connect to ARC so maybe
the account isn't working yet. 

To do:
Find out how to load models with custom layers and functions
Data augmentation ( from keras.preprocessing.image import ImageDataGenerator) )
Revise scheduler
Unit testing, containerization, turning code into package
Determine the actual experiments/training to be done on ARC

---

Reminder for Duncan: 

git add . 

git commit -m "_message_" 

git push origin main 

git pull origin main

pip3 list – format=freeze > requirements.txt


