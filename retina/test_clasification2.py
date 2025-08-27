''' script to test clasification'''
#%%
import numpy as np
import napari
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
#import glasbey
from skimage.color import hsv2rgb, rgb2hsv

ffolder = r'G:\office\work\git\retina\DATA\2018004_1_1'
imageFile = '2018004_71_960nm_116fr_3Pro_1_1-Ch2-_intensity_image.tif'
tauFile = '2018004_1_1_tm_Ch2.xlsx'
#classFile = '2018004_1_1_ch2_classification.ods'
#columnName = 'classification'
classFile = '2018004_1_1_ch2_classification2.xlsx'
columnName = 'classification_Hala'

labelFile = '2018004_1_1_ch2_labels.txt'

class GrainId:
    name = {'BACKGROUND': 0,
            'L': 1,
            'M': 2,
            'ML':3,
            'NON':4}

classColorNP = np.array(['black', 'green','blue','red','black'])
myColorMap = {1:[0,1,0],2:[0,0,1],3:[1,0,0], 4:[0,0,0]}



#%% data loading

image = tifffile.imread(ffolder +'/' + imageFile)
tau = pd.read_excel(ffolder +'/' + tauFile, sheet_name="TauMean 1", header=None).to_numpy()[::-1,:]

_sheet = pd.read_excel(ffolder +'/' + classFile, sheet_name="Sheet1")
gPos = np.vstack((_sheet['X'].to_numpy(),_sheet['Y'].to_numpy())).T.astype(int)

_gClass = _sheet[columnName].to_list()
gClass = np.array([GrainId.name[str.upper(ii)] for ii in _gClass])

givenLabel = pd.read_csv(ffolder +'/' + labelFile, sep='\t').to_numpy()

classImage = np.zeros_like(image)
classImage[gPos[:,-1],gPos[:,-2]] = gClass


# tauIntensity Image
tauScaled = (tau-50)/1000
_intTauImage = np.array((tauScaled,np.ones_like(image),image/np.max(image)))
intTauImage = np.swapaxes(np.swapaxes(hsv2rgb(_intTauImage, channel_axis = 0),0,2),0,1)


#%% show original data

# original glasbey
#glas = glasbey.create_palette(256)

viewer = napari.Viewer()

#viewer.add_image(image)
#viewer.add_labels(givenLabel, colormap=glas)
#viewer.add_image(tau)
viewer.add_image(intTauImage, rgb=True)
viewer.add_labels(classImage,colormap=myColorMap)


#%% global threshold of the fluorescence signal
from skimage import data, restoration, util
from skimage.filters import threshold_otsu
import skimage as ski
from skimage.filters import sobel


thresh = ski.filters.threshold_otsu(image)
binary = image > thresh

# add extra darker area in necessary
#binary = image > thresh*0.5

#viewer.add_image(binary)

#%% blob detection
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.draw import disk
from skimage.segmentation import expand_labels


blobs_log = blob_log(image*binary, max_sigma=5, min_sigma=2, overlap=0.5, threshold=0.0001)
labels_data = np.zeros_like(image)
nBlob = len(blobs_log[:])
for ii,_blob in enumerate(blobs_log):

    rr, cc = disk((_blob[0], _blob[1]), _blob[2], shape=image.shape)

# Assign a label value (e.g., 1) to the circular area
    label_value = ii+1
    labels_data[rr, cc] = label_value

expanded = expand_labels(labels_data, distance=1)

#viewer.add_labels(labels_data, colormap=glas)
#viewer.add_labels(expanded, name='blob')


#%% set ground true labels to my labels

_myClass = np.zeros(nBlob+1,dtype=int)
_myClass[expanded[classImage>0]] = classImage[classImage>0].astype(int)
#set zero index to background 
_myClass[0] = 0

myClass = _myClass[1:]

# get the image of blobs with ground true classification
myClassImage = np.zeros_like(image)
myClassImage = _myClass[expanded]

#viewer.add_labels(myClassImage, colormap=myColorMap)

#%% get the clarification parameters
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

distance = ndi.distance_transform_edt(expanded>0)
#viewer.add_image(distance, name='distanced')

# nPoly ... fitting polynomica order
nPoly = 2
# nR ... resolution in radial distance of the spot
nR = 50

radius = np.linspace(0,1,nR)
tauFit = np.zeros((nBlob,nR))
intFit = np.zeros((nBlob,nR))

#fig1, ax1 = plt.subplots()
#fig2, ax2 = plt.subplots()


for ii in range(nBlob):
    mask = expanded==ii+1
    intIdx = image[mask]
    tauIdx = tau[mask]

    _disIdx = distance[mask]
    disMax = np.max(_disIdx)
    disIdx = (disMax - _disIdx)/(disMax-1)
    
    # intensity fit
    z = np.polyfit(disIdx, intIdx, nPoly)
    intFit[ii,:] = np.poly1d(z)(radius)

    #ax1.scatter(disIdx.ravel(),intIdx.ravel(),color = classColorNP[myClass[ii]])
    #ax1.plot(radius,intFit[ii,:],color = classColorNP[myClass[ii]])

    # tau fit
    z = np.polyfit(disIdx, tauIdx, nPoly)
    tauFit[ii,:] = np.poly1d(z)(radius)

    #ax2.scatter(disIdx.ravel(),tauIdx.ravel(),color = classColorNP[myClass[ii]])
    #ax2.plot(radius,tauFit[ii,:], color = classColorNP[myClass[ii]])

maxInt = np.max(intFit,axis=1)
maxTau = np.max(tauFit,axis=1)
ratioInt = intFit[:,0]/np.max(intFit,axis=1)
ratioTau = tauFit[:,-1]/tauFit[:,0]

#%% plot the fits

fig1, ax1 = plt.subplots()
for ii,_color in enumerate(classColorNP):
    ax1.plot(radius,intFit[myClass==ii,:].T,color=classColorNP[ii])
ax1.set_title('intensity profile')
ax1.set_ylabel('intensity /a.u.')
ax1.set_xlabel('radius ')

fig1, ax1 = plt.subplots()
for ii,_color in enumerate(classColorNP):
    ax1.plot(radius,tauFit[myClass==ii,:].T,color=classColorNP[ii])
ax1.set_title('tau profile')
ax1.set_ylabel('tau /ns')
ax1.set_xlabel('radius ')


# some correlation graph
#fig3, ax3 = plt.subplots()
#ax3.scatter(myClass,ratioTau, color= 'blue')
#ax3.scatter(myClass,ratioInt, color= 'red')
#ax3.scatter(myClass,np.max(intFit,axis=1), color= 'red')
#ax3.scatter(myClass,np.max(tauFit,axis=1), color= 'red')
#ax3.scatter(maxTau,ratioInt,s=20, color = classColorNP[myClass])

# %% set classification parameters classification

medTau = np.median(maxTau)
medInt = np.median(maxInt)
thrTau = 0.8
thrInt = 0.75
thrTauRatio = 1.15

#print(f'thrTau {medTau*thrTau} ns')
#print(f'thrInt {medInt*thrInt} a.u.')

fig3, ax3 = plt.subplots()
ax3.scatter(maxTau,maxInt,s=20, color = classColorNP[myClass])
ax3.scatter(medTau,medInt, s= 40, color = 'white')
ax3.hlines(medInt*thrInt, np.min(maxTau),np.max(maxTau), linestyles= ':')
ax3.vlines(medTau*thrTau, np.min(maxInt),np.max(maxInt),linestyles= ':')
ax3.set_title('Classification criteria  Melanin / Lipofuscin')
ax3.set_ylabel('intensity /a.u.')
ax3.set_xlabel('tau / ns ')
ax3.annotate('M', xy= (medTau*thrTau*0.9,medInt*thrInt*0.9))

fig3, ax3 = plt.subplots()
ax3.scatter(ratioTau,ratioInt,s=20, color = classColorNP[myClass])
#ax3.hlines(medInt*thrInt, np.min(maxTau),np.max(maxTau), linestyles= ':')
ax3.vlines(thrTauRatio, np.min(ratioInt),np.max(ratioInt),linestyles= ':')
ax3.set_title('Classification criteria MelanoLipofuscin')
ax3.set_ylabel('Int_centre / Int_max ')
ax3.set_xlabel('Tau_edge / Tau_centre ')
ax3.annotate('ML', xy= (thrTauRatio,1))


# according the criteria classify my clusters
# set everything to Lipofuscin
myFitClass = np.zeros_like(myClass)
myFitClass[:] = GrainId.name['L']
# select melanin
myFitClass[((maxTau< medTau*thrTau) & (maxInt < medInt*thrTau))] = GrainId.name['M']
# identify melanolipofuscin
myFitClass[ratioTau > thrTauRatio] = GrainId.name['ML']


_myFitClass= np.hstack(([0],myFitClass))
myClassFitImage = np.zeros_like(image)
myClassFitImage = _myFitClass[expanded]

viewer.add_labels(myClassFitImage, colormap=myColorMap)


plt.show()


# %%
