

'''Classi4RPE is a computational program to segment and classify
 the granules of Retinal Pigment Epithelium cells RPE
 
 Created in 2025
 Wrtitten by: Ondrej Stranik, Maryam Ali, Rainer Heintzmann
 
 It can read FLIM and intensity data for RPE measurements, and:
     - segment the granules after thresholding short/long lifetimes using seeded water shedding.
     - Identify Lipofuscins (Higher fluorescent)
     - Identify lower fluorescent granules and distiguish Malanolipouscins by computing their
     lifetime ratio from center to edge.
     - Export the segmented & classified granules data: coordinates, mean lifetimes.
     - Visualize selectied granules interactively by selecting the lifetime/intensity range from the histogram.
 
 
 '''

#%% Imports

from datetime import date, time
import pandas as pd
from tkinter import filedialog
import numpy as np
import tifffile
import napari
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from scipy import ndimage
from skimage.segmentation import find_boundaries


from Functions_Classi4RPE import *


#%% Importing files
# Browsing files by order of: Intensity Image, then ascii files for the fitting (12 asciis), and filnally sdt

file_list = []

print('Please select the files by this order: Intensity Image, 12 ascii fitting files, and sdt')


tif_file = filedialog.askopenfilenames(title = 'Please select the intensity Image', filetypes = (('Tif Image', '*.tif*'), 
                                                                                         ('All files', '*.*')))

asc_file = filedialog.askopenfilenames(title = 'Please select asci files', filetypes = (('ASCII', '*.asc*'),
                                                                                         ('All files', '*.*')))

sdt_file = filedialog.askopenfilenames(title = 'Please select sdt file', filetypes = (('Sdt', '*.sdt*'), 
                                                                                         ('All files', '*.*')))

file_list.append(tif_file)
file_list.append(asc_file)
file_list.append(sdt_file)

print('you selected those:')
for f in file_list:
    print(f)

#%% For ground truth data reading

#ffolder = r'C:\Users\Maryam\Documents\Granules Classification\Test_seg_class\2018004_1_1_ch2'
#Ground truth classification file
#classFile = '2018004_1_1.xlsx'
#columnName = 'classification'

#classImage = loadClassification(ffolder,classFile,imageSize= image.shape, columnName= columnName)

#%% Reading the data

tau1, tau2, image, sdt_d, data_name = Import_data(file_list)



#%% Processing the lifetime values with the intensity image (Visualize the lifetime image)
# and Plotting

intTauImage = getTauIntensityImage(image,tau2)

viewer = napari.Viewer()
viewer.add_image(image, name='intensity')
viewer.add_image(intTauImage, rgb=True)
#viewer.add_labels(classImage,colormap = GrainId.colorNapari)

#%% 2D histogram tau x intensity

H, xEdge, yEdge, binNumber = getTauIntensityHistogram(image,tau2)

#fig3, ax3 = plt.subplots()
#ax3.pcolormesh(xEdge, yEdge, H, cmap='rainbow')
#ax3.set_title('Tau Intensity pixels')
#ax3.set_ylabel('intensity /a.u.')
#ax3.set_xlabel('tau / ns ')

#%% Interactive visualization of selected area in the histogram


#showSelectedArea(H, binNumber, intTauImage)


#%% Setting a threshold to distinguish between long/short lifetime granules
#threshold has been optimized based on the tested data sets
# and it can be selected upon the lifetime values


tau = np.nan_to_num(tau2) 
tau1 = np.nan_to_num(tau1)    
tau_thresh = (np.percentile(tau[tau>0], 15)) +12

#get mask for M and L 

lipImage = getTauRangeImage(image,tau, tauRange=[tau_thresh+1,tau.max()])

lipBinary = getMask(lipImage)

melImage = getTauRangeImage(image,tau, tauRange=[0, tau_thresh])

melBinary = getMask(melImage)



#%% segmentation of granules (long lifetimes and shortlifetimes)

# By seeded water shedding
# expansion was optimized based on the tested data sets (to reasonably matching the exact size of the granules)

melLabel = seeded_water_shed(melImage*melBinary, min_distance = 7, expansion=2)
lipLabel = seeded_water_shed(lipImage*lipBinary, min_distance = 3, expansion = 1)

# Optional: another segmentation approach (Blobs)
#melLabel, _nBlob1 = getGranule(melImage,melBinary,min_sigma=2, max_sigma=4, extraExpansion=2)
#lipLabel, _nBlob2 = getGranule(lipImage,lipBinary,min_sigma=3, max_sigma=5)


_nBlob1 = len(np.unique(melLabel))
_nBlob2 = len(np.unique(lipLabel))
#viewer.add_labels(melLabel)
#viewer.add_labels(lipLabel)


max_M = melLabel.max()
lipLabel_shifted = np.where(lipLabel > 0, lipLabel + max_M, 0)  
Overview_map = np.maximum(melLabel, lipLabel_shifted)

M_class = np.zeros((melLabel.max()),dtype=int) +4

# segmented image of the all segments without expansion

melLabel_noex = seeded_water_shed(melImage*melBinary, min_distance = 7, expansion=0)
lipLabel_noex = seeded_water_shed(lipImage*lipBinary, min_distance = 3, expansion = 1)

max_M_noex = melLabel_noex.max()
lipLabel_shifted_noex = np.where(lipLabel_noex > 0, lipLabel_noex + max_M_noex, 0)  
segments_noex = np.maximum(melLabel_noex, lipLabel_shifted_noex)
borders_noex = (find_boundaries(segments_noex, mode='outer'))



#%% Comparing the classification with ground truth 


#melGTImage, melGTClass = projectGroundTrueToLabels(classImage, melLabel)
#lipGTImage, lipGTClass = projectGroundTrueToLabels(classImage, lipLabel)

#GTImage = np.max(np.array([melGTImage, lipGTImage]), axis=0)

#g_Image, g_Class = projectGroundTrueToLabels(classImage, Overview_map)


#viewer.add_labels(GTImage, colormap=GrainId.colorNapari)

#%% Get intensity/tau profiles of the short lifetime granules (Mlanolipofuscins / Melanin)
#Apply tau fitting for each granule(segment) and plot the profile

radius, intFit, tauFit = getProfiles(image, tau,melLabel, nPoly=2)

#fig, ax = plt.subplots()

#for ii,_color in enumerate(GrainId.colorMPL):
#    ax.plot(radius,tauFit[M_class==ii+1,:].T)
#    ax.set_title('tau profile')
#    ax.set_ylabel('tau /ps')
#    ax.set_xlabel('radius ')


#%% Plot values of tau & intensity ratios for each segment
#between edge to center

maxInt = np.max(intFit,axis=1)
maxTau = np.max(tauFit,axis=1)
ratioInt = intFit[:,0]/maxInt
ratioTau = tauFit[:,-1]/tauFit[:,0]


#Set a threshold for distinguishing Melanin from Melanolipofuscins
# this threshold has been optimized based on the tested data sets
thrTauRatio = 1.15


#fig, ax = plt.subplots()

#ax.scatter(ratioTau,ratioInt,s=20, color = GrainId.colorMPL[M_class-2])
#ax.vlines(thrTauRatio, np.min(ratioInt),np.max(ratioInt),linestyles= ':')
#ax.set_title('Classification criteria MelanoLipofuscin')
#ax.set_ylabel('Int_centre / Int_max ')
#ax.set_xlabel('Tau_edge / Tau_centre ')
#ax.annotate('ML', xy= (thrTauRatio,1))

#%% according the criteria classify ML clusters from M
melFitClass = separateMLfromM(tauFit, thrTauRatio=thrTauRatio)

#%% Combine L & ML classification
# & Plotting

melFitImage = myClassToImage(melFitClass,melLabel_noex)
lipFitImage = (lipLabel>0)*GrainId.name['L']

# add L M and ML together in one image
allFitImage = np.max(np.array([melFitImage, lipFitImage]), axis=0)

#viewer.add_labels(allFitImage, colormap=GrainId.colorNapari)

# create a visual image based on the 'not' expand segmentation, for better visualization
# borders are added to see the segments clearly 
visual_classImage = allFitImage + (borders_noex * 4)
classi_layer = viewer.add_labels(visual_classImage, name='Classi', colormap = GrainId.colorNapari)

#%% Finetuning & manual modifications on the classification
#By clicking on the sgmented image then a => L, q => M, j => ML
#

L_tuned, M_tuned, ML_tuned, = tune_class_click(Overview_map, allFitImage, visual_classImage, classi_layer, viewer)


#%% assigned the finetuning to the image & save it


Tuned_classImg_vis = visual_classImage.copy() 
Tuned_classImg =  visual_classImage.copy()  

Tuned_classImg[np.isin(Tuned_classImg, L_tuned)] = 1
Tuned_classImg[np.isin(Tuned_classImg, M_tuned)] = 2
Tuned_classImg[np.isin(Tuned_classImg, ML_tuned)] = 3

Tuned_classImg_vis[np.isin(Tuned_classImg_vis, L_tuned)] = 1
Tuned_classImg_vis[np.isin(Tuned_classImg_vis, M_tuned)] = 2
Tuned_classImg_vis[np.isin(Tuned_classImg_vis, ML_tuned)] = 3


for layer in viewer.layers:
    layer.visible = False
classi_layer.visible = True
viewer.screenshot('classiImage.tif')


#%% Extract predictions of classifications

labels = np.unique(Overview_map)


class_predictions = {lbl: np.unique(Tuned_classImg[Overview_map == lbl])[0] for lbl in labels}
predictions = np.array([class_predictions[lbl] for lbl in sorted(class_predictions)])[1:]


#%%Sensitivity & Specificity
'''

grs = [1, 2, 3, 4]

Evaluation = sensitivity_specificity(g_Class, predictions, labels=grs)
print("with not identified granules: " )
print(Evaluation)

# Removing the not identified granules from the claculation of: Sensitivity & Specificity

print("***")
print("***")

gt = []
not_classified = []

for s in range(len(g_Class)):
    if g_Class[s] == 1:
        gt.append("1")
    if g_Class[s] == 2:
        gt.append("2")
    if g_Class[s] == 3:
        gt.append("3")
    if g_Class[s] == 4:
        gt.append("NON")
        not_classified.append(s)
        

gt = np.array(gt)
non = np.array(not_classified)

g_Class_non = np.delete(g_Class, not_classified[1:])
predictions_non = np.delete(predictions, not_classified[1:])

grs_non = [1, 2, 3]

Evaluation_non = sensitivity_specificity(g_Class_non, predictions_non, labels=grs_non)
print("Excluding the un-identified granules: ")
print(Evaluation_non)


    
'''

#%% Data to be exported

gr_num = Overview_map.max()


centers = []
mean_tau_ch2 = []
mean_tau_ch1 = []
no_photons1 = []
no_photons2 = []

#Granule's size: no of pixels
no_pixels = (ndimage.sum(np.ones_like(Overview_map), Overview_map, index=np.unique(Overview_map)))[1:]


for i in (np.unique(Overview_map)[1:]):
    segment = (Overview_map==i)
    cen_of_mass = ndimage.center_of_mass(segment)
    centers.append(cen_of_mass)
    Lifetau1 = tau1[segment]
    mean_tau1 = np.mean(Lifetau1)
    mean_tau_ch1.append(mean_tau1)
    Lifetau2 = tau[segment]
    mean_tau2 = np.mean(Lifetau2)
    mean_tau_ch2.append(mean_tau2) 
    #number of photons per granule
    photons_ch1 = np.sum((np.sum(sdt_d[0, :, :, :], axis = 2))[segment])
    photons_ch2 = np.sum((np.sum(sdt_d[1, :, :, :], axis = 2))[segment])
    no_photons1.append(photons_ch1)
    no_photons2.append(photons_ch2)


    
coord = np.array(centers)
meanT2 = np.array(mean_tau_ch2)
meanT1 = np.array(mean_tau_ch1)
photons1 = np.array(no_photons1)
photons2 = np.array(no_photons2)


seg_labeled = np.arange(1, (Overview_map.max()+1))
empty = np.zeros((seg_labeled.shape))
dataID = np.full((seg_labeled.shape), [data_name])

Export_results = {"Data ID":dataID, "segment":seg_labeled, "Class [L=1, M=2, ML=3]":predictions,
                  "X":coord[:, 1], "Y":coord[:, 0],
                  "mean_tau_Ch1":meanT1, "mean_tau_Ch2":meanT2,
                  "no. of pixels":no_pixels, "no. of photons SSC":photons1, "no. of photons LSC":photons2, 
                  "Peak Emmision Wavelength":empty}
Export = pd.DataFrame(Export_results)

Export.to_excel("Results_"+ str(data_name) + ".xlsx", index = False)



