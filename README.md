# Classi4RPE


Classi4RPE is a computational program to segment and classify the granules of Retinal Pigment Epithelium cells RPE
this classification is based on the Fluorescence lifetime measurements
 
Created in 2025
Wrtitten by: Maryam Ali, Ondrej Stranik,  Rainer Heintzmann
Used for study: Ali, M., Alhaj Ahmad, H., Alderzy, H., Hammer, M., Heintzmann, R., & Stranik, O. (2026). Segmentation and classification of retinal pigment granules in fluorescence lifetime imaging microscopy (FLIM) data (Version 1). bioRxiv. https://doi.org/10.64898/2026.06.29.735375


It can read FLIM and intensity data for RPE measurements, and:
   - segment the granules after thresholding short/long lifetimes using seeded water shedding.
   - Identify Lipofuscins (Higher fluorescent)
   - Identify lower fluorescent granules and distiguish Malanolipouscins by computing their
   lifetime ratio from center to edge.
   - Export the segmented & classified granules data: coordinates, mean lifetimes.
   - Visualize selectied granules interactively by selecting the lifetime/intensity range from the histogram.
   - Finetune the classification directly via Napari GUI.

Classi4RPE_GUI: is a fully simple GUI for importing data files and process them, allowing the user to finetune the data and change the classification.
Imported data should contain: Intensity image, fitting ascii files for lifetime, and sdt data. 

tested data includes an example of tested data.
other data sets, which have been used to set the parameters are published on: https://doi.org/10.5281/zenodo.20702171 

This code (including the setted parameters) has been tested on FLIM data sets from University Hospital Jena, Experimental Ophthalmology Group using Becker
& Hickl GmbH.


 
