Duncan Boyd, Aug 19, 2022
duncan@wapta.ca

Description: This repository is for all code written as a summer student in the Seaman MRI Center 2022. 
Includes debugging programs and ARC programs to allow a user to compare complex vs. real valued algebra in u-nets.
It is not particularily optimized, and could be structured better. 

Note: Actual complex algebra (one of the only completely original parts of this code) can
be found in unet_compare/functions line 366. The rest of the code is support to test this. 

Note: Code was greatly inspired (copied) from Dr. Frayne and Dr. Souza's code (link below).
This is turn from their paper: 
Souza, Roberto, and Richard Frayne. "A hybrid frequency-domain/image-domain deep network for magnetic resonance image reconstruction." In 2019 32nd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), pp. 257-264. IEEE, 2019.

Github link: https://github.com/rmsouza01/Hybrid-CS-Model-MRI

---

Status: Project finished.

---

Structure:

arc_test, local_test and pred_test are main scripts. Desciption on what they do are commented at the top of each file.

unet_compare provides support for these three functions.

Most other files are either inputs, outputs, debugging or utilities.





