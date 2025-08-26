''' script to test clasification'''
#%%
import numpy as np
import napari
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
import glasbey
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
glas = glasbey.create_palette(256)

viewer = napari.Viewer()

viewer.add_image(image)
#viewer.add_labels(givenLabel, colormap=glas)
#viewer.add_image(tau)
viewer.add_image(intTauImage, rgb=True)
viewer.add_labels(classImage,colormap=glas)


#%% global threshold of the fluorescence signal
from skimage import data, restoration, util
from skimage.filters import threshold_otsu
import skimage as ski
from skimage.filters import sobel


thresh = ski.filters.threshold_otsu(image)
binary = image > thresh
#viewer.add_image(binary)

#%% remove local low fluorescence signal 
from skimage.morphology import binary_erosion, binary_closing, binary_dilation

block_size = 35
local_thresh = ski.filters.threshold_local(image, block_size, offset=0)
binary_local = image > local_thresh
finerBinary = binary_local*binary
#viewer.add_image(finerBinary)


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
viewer.add_labels(expanded,colormap=glas, name='blob')


#%% true labels to my labels
#TODO: correct the assigment!!
idxToClass = np.vstack((expanded[classImage>0],classImage[classImage>0]))

myClass = np.zeros(nBlob,dtype=int)
myClass[idxToClass[0,:]+1] = idxToClass[1,:].astype(int)

myClassImage = np.zeros_like(image)
# create just points
#myClassImage[blobs_log[:,0].astype(int),blobs_log[:,1].astype(int)] = myClass

# create patches
for ii in np.arange(nBlob):
    myClassImage[expanded==ii+1] = myClass[ii]

viewer.add_labels(myClassImage, colormap=glas)


#%% get the clarification parameters
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
distance = ndi.distance_transform_edt(expanded>0)
#viewer.add_image(distance, name='distanced')

classColor = {1: 'None', 2: 'green', 3: 'blue', 4:'black', 0:'black'}
classColorNP = np.array(['black', 'green','blue','red','black'])


nPoly = 2
nR = 50
radius = np.linspace(0,1,nR)
tauFit = np.zeros((nBlob,nR))
intFit = np.zeros((nBlob,nR))

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()


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

    #ax1.scatter(disIdx.ravel(),intIdx.ravel(),color = classColor[myClass[ii]])
    ax1.plot(radius,intFit[ii,:],color = classColor[myClass[ii]])

    # tau fit
    z = np.polyfit(disIdx, tauIdx, nPoly)
    tauFit[ii,:] = np.poly1d(z)(radius)

    #ax2.scatter(disIdx.ravel(),tauIdx.ravel(),color = classColor[myClass[ii]])
    ax2.plot(radius,tauFit[ii,:], color = classColor[myClass[ii]])

maxInt = np.max(intFit,axis=1)
maxTau = np.max(tauFit,axis=1)
ratioInt = intFit[:,0]/np.max(intFit,axis=1)
ratioTau = tauFit[:,-1]/np.max(tauFit,axis=1)

print(f'intensity ratio {ratioInt}')
print(f'tau ratio {ratioTau}')

#%%
fig3, ax3 = plt.subplots()

#ax3.scatter(myClass,ratioTau, color= 'blue')

#ax3.scatter(myClass,ratioInt, color= 'red')

#ax3.scatter(myClass,np.max(intFit,axis=1), color= 'red')

#ax3.scatter(myClass,np.max(tauFit,axis=1), color= 'red')

ax3.scatter(maxTau,ratioInt,s=20, color = classColorNP[myClass])



