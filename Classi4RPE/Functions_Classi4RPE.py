''' 
Functions used for Classi4RPE
'''

#%%
import numpy as np
import napari
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
from sdtfile import SdtFile
import os
from pathlib import Path
#import glasbey
from skimage.color import hsv2rgb, rgb2hsv
from scipy.stats import binned_statistic_2d
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_closing, binary_dilation
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.draw import disk
from skimage.segmentation import expand_labels
from scipy.special import erf
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.color import label2rgb
from scipy.ndimage import  generate_binary_structure
from sklearn.metrics import confusion_matrix
from skimage.segmentation import relabel_sequential
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from skimage.segmentation import find_boundaries




#%% functions

class GrainId:
    name = {'BACKGROUND': 0,
            'L': 1,
            'M': 2,
            'ML':3,
            'NON':4}
    colorMPL = np.array(['red', 'green','blue','white'])
    colorNapari = {1:[1,0,0],2:[0,1,0],3:[0,0,1], 4:[1,1,1]}


def loadData(ffolder, imageFile,tauFile1, tauFile2):
    # load measured data
    image = tifffile.imread(ffolder +'/' + imageFile)
    tau1 = pd.read_excel(ffolder +'/' + tauFile1, sheet_name="TauMean 1", header=None).to_numpy()[::-1,:]
    tau2 = pd.read_excel(ffolder +'/' + tauFile2, sheet_name="TauMean 1", header=None).to_numpy()[::-1,:]
    
    #remove nan data
    tau1[np.isnan(tau1)] = 0
    tau2[np.isnan(tau2)] = 0

    return image, tau1, tau2

def loadClassification(ffolder, classFile, imageSize, columnName= 'classification_Hala', sheet_name="Sheet1"):
    # load ground true
    _sheet = pd.read_excel(ffolder +'/' + classFile, sheet_name=sheet_name)
    gPos = np.vstack((_sheet['X'].to_numpy(),_sheet['Y'].to_numpy())).T.astype(int)

    _gClass = _sheet[columnName].to_list()
    gClass = np.array([GrainId.name[str.upper(ii.replace(" ", ""))] for ii in _gClass])

    classImage = np.zeros(imageSize,dtype=int)
    #classImage[imageSize[0] - gPos[:,-1],gPos[:,-2]] = gClass
    #classImage[ gPos[:,-1],gPos[:,-2]] = gClass
    classImage[gPos[:,-2],gPos[:,-1]] = gClass


    return classImage

def loadMLabel(ffolder, labelFile):
    # load Maryam segmentation
    givenLabel = pd.read_csv(ffolder +'/' + labelFile, sep='\t').to_numpy()
    return givenLabel


def getTauIntensityImage(image,tau,tauRange= [50,1000]):
    # get tauIntensity Image
    tauScaled = (tau-tauRange[0])/tauRange[1]
    tauScaled[tauScaled<0] = 0
    tauScaled[tauScaled>1000] = 1

    _intTauImage = np.array((tauScaled,np.ones_like(image),image/np.max(image)))
    intTauImage = np.swapaxes(np.swapaxes(hsv2rgb(_intTauImage, channel_axis = 0),0,2),0,1)
    return intTauImage


def getTauIntensityHistogram(image,tau, tauRange=[50,1000], bins=200):
    ''' get 2D histogram'''
    _tau = np.copy(tau)
    # remove nan value
    _tau[np.isnan(_tau)] = 0

    sel = (tau<tauRange[0]) | (tau>tauRange[1])
    _tau[sel] = 0
    _tau = _tau.flatten()
    _image = np.copy(image)
    _image = _image.flatten()

    H, xEdge, yEdge, binnumber = binned_statistic_2d(
    _tau,_image, None, bins=bins, range = [tauRange,[0, np.max(_image)]],statistic='count',expand_binnumbers=True)
    H = H.T
    return H, xEdge, yEdge, binnumber

def getMask(image, extra_factor=1, minSize=3):
    #get mask
    thresh = threshold_otsu(image)
    binary = image > thresh*extra_factor

    binary = binary_erosion(binary, footprint= np.ones((minSize, minSize)))
    binary = binary_dilation(binary, footprint= np.ones((minSize, minSize)))

    return binary

def getMaskOnTau(tau,tauRange, sigma= 3):
    ''' create mask from tau. apply smooth band pass filter
     return np.array of size tau, values from 0 to 1  '''
    fact1 = (erf((tau-tauRange[0])/sigma)+1)/2
    fact2 = (erf((tauRange[1]-tau)/sigma)+1)/2
    return fact1*fact2

def getTauRangeImage(image,tau,tauRange=[20,200], sigma=3):
    # get intensity image for certain tau range
    return image*getMaskOnTau(tau,tauRange)

def showSelectedArea(H, binNumber, intTauImage):
    ''' create two interactive napari viewer showing selected area of histogram in image'''

    def updateImage():
        print('image updated')
        try:
            _label = viewer2.layers['area'].to_labels(H.shape)
            viewer2.layers['labels'].data = _label
            shapeIdx = binNumber.reshape((2,*intTauImage.shape[:2]))
            idxInImage = _label[shapeIdx[1,...]-1,shapeIdx[0,...]-1]
            viewer.layers['idxInImage'].data = idxInImage
        except:
            print('not updated')

    viewer = napari.Viewer()
    viewer.add_image(intTauImage, rgb=True)
    viewer.add_labels(np.zeros(intTauImage.shape[:2], dtype=int), name='idxInImage')

    viewer2 = napari.Viewer()
    viewer2.add_image(H, name='2D-histogram', colormap='hsv')
    viewer2.add_shapes(name= 'area')
    viewer2.add_labels(np.zeros_like(H, dtype=int),name= 'labels')
    viewer2.layers['area'].events.data.connect(updateImage)


def getGranule(image,binary,max_sigma=5, min_sigma=2, 
               overlap=0.5, threshold=0.1,labelOffset=0, extraExpansion=0):
    ''' get labelled mask of the granule
    assumed granules are spherical mit min max radius'''
    # get position of granule and its size
    blobs_log = blob_log(image*binary, 
                         max_sigma=max_sigma, 
                         min_sigma=min_sigma, 
                         overlap=overlap,
                         threshold=threshold,
                         exclude_border=True)
    labels_data = np.zeros_like(image)
    nBlob = len(blobs_log[:])
    for ii,_blob in enumerate(blobs_log):
        rr, cc = disk((_blob[0], _blob[1]), _blob[2], shape=image.shape)
        labels_data[rr, cc] = ii+1 + labelOffset  

    # extra expand the label by one
    labelImage = expand_labels(labels_data, distance=extraExpansion).astype(int)

    return labelImage, nBlob

def projectGroundTrueToLabels(classImage,labelImage):
    ''' project ground true classes to the labels'''
    nBlob = np.max(labelImage)
    expanded_label_img = expand_labels(labelImage, 0)
    _myClass = np.zeros(nBlob+1,dtype=int) +4
    #_myClass[labelImage[classImage>0]] = classImage[classImage>0].astype(int)
    _myClass[expanded_label_img[classImage>0]] = classImage[classImage>0].astype(int)
    #set zero index to background 
    _myClass[0] = 0
    # get the image of blobs with ground true classification
    myClassImage = np.zeros_like(labelImage)
    myClassImage = _myClass[labelImage]

    # remove background index
    myClass = _myClass[1:]

    return myClassImage, myClass

def getProfiles(image,tau,expanded, nPoly=2, showData= False, myClass=None):
    ''' nPoly ... order of the polynomial fit'''
    distance = ndi.distance_transform_edt(expanded>0)
    #viewer.add_image(distance, name='distanced')

    # nR ... resolution in radial distance of the spot
    nR = 50

    nBlob = np.max(expanded)

    radius = np.linspace(0,1,nR)
    tauFit = np.zeros((nBlob,nR))
    intFit = np.zeros((nBlob,nR))

    if showData:
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

        # tau fit
        z = np.polyfit(disIdx, tauIdx, nPoly)
        tauFit[ii,:] = np.poly1d(z)(radius)


        if showData:
            ax2.plot(radius,tauFit[ii,:], color = GrainId.colorMPL[myClass[ii]-1])

    return radius, intFit, tauFit

def separateMLfromM(tauFit, thrTauRatio = 1):
    ''' separate ML from M according the tau profile'''
    maxTau = np.max(tauFit,axis=1)
    ratioTau = tauFit[:,-1]/tauFit[:,0]

    # set everything to M
    myFitClass = np.zeros_like(maxTau,dtype=int)
    myFitClass[:] = GrainId.name['M']

    myFitClass[ratioTau > thrTauRatio] = GrainId.name['ML']

    return myFitClass

def myClassToImage(myFitClass, labelImage):
    ''' make image of granule colored with a classification'''
    _myFitClass= np.hstack(([0],myFitClass))
    myClassFitImage = np.zeros_like(labelImage)
    myClassFitImage = _myFitClass[labelImage]

    return myClassFitImage

def seeded_water_shed(img, min_distance = 4, expansion = 0):
    d1 = gaussian_filter(img, 1.6)
    d2 = gaussian_filter(img, 3.0)
    dog = d1 - d2
    dog = dog / dog.max()
    
    local_max = peak_local_max(dog, min_distance=min_distance, threshold_abs=0.01)
    mask_local = np.zeros_like(dog, dtype=bool)
    mask_local[tuple(local_max.T)] = True
    seeds = mask_local & (dog > 0.1)
    markers, num_features = label(seeds)
    
    connectivity = generate_binary_structure(2, 2)
    water_shed = watershed(image = -dog, markers = markers, mask = (dog>0.1), connectivity = connectivity)
    
    expand_wt = expand_labels(water_shed, distance=expansion).astype(int)
    
    L, counts = np.unique(expand_wt, return_counts=True)
    for i in np.unique(L):
        if counts[i] <10:
            expand_wt[expand_wt==i] = 0
    
    expand_wt, _, _ = relabel_sequential(expand_wt)
    
    return expand_wt

def sensitivity_specificity(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    results = {}

    for i, cls in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        results[cls] = {"sensitivity": sensitivity, "specificity": specificity}

    return results


def read_asc(channel_data):
    data = []
    for j in range(6):
        with open(channel_data[j]) as f:
            ch_data = ([line.strip() for line in f if line.strip()])
            f_asc = np.array([np.fromstring(x.strip("[]"), sep=' ') for x in ch_data], dtype=float)[::-1,:]
            data.append(f_asc)
    channel_arrays = np.array(data)
    tau_asc_v = (channel_arrays[0] * channel_arrays[3]) + (channel_arrays[1] * channel_arrays[4]) + (channel_arrays[2] * channel_arrays[5])
    alpha = channel_arrays[0] + channel_arrays[1] + channel_arrays[2]
    tau_asc = np.divide(tau_asc_v, alpha, out = np.zeros_like(tau_asc_v, dtype=float), where=alpha!=0)
    return tau_asc



def Import_data(file_list):
    # Identify data name
    b = os.path.basename(file_list[0][0])
    b_name = Path(b).stem
    data_name_1 = (b_name).replace("_71_960nm_116fr_3Pro","")
    data_name = (data_name_1).replace("-Ch2-_intensity_image", "")
    
    #Intensity Image
    image = tifffile.imread(file_list[0])
    
    #sdt file
    s_sdt = (file_list[2][0]).rstrip(",")
    data_sdt = SdtFile(s_sdt)
    sdt = np.array(data_sdt.data)
    
    # Ascii files
    ch1_data = [file_list[1][0], file_list[1][1],
              file_list[1][2], file_list[1][3],
              file_list[1][4], file_list[1][5]]
    
    ch2_data = [file_list[1][6], file_list[1][7],
              file_list[1][8], file_list[1][9],
              file_list[1][10], file_list[1][11]]
    
    tau1 = read_asc(ch1_data)
    tau2 = read_asc(ch2_data)

    
    return tau1, tau2, image, sdt, data_name

    
    
def tune_class_click(labels, old_classImage, old_visualImage, classi_layer, view_layer):
    
    segments = view_layer.add_labels(labels, name="segments")


    selected_labels = set()
    last_clicked_label = [None]   

    L_tuned = []
    M_tuned = []
    ML_tuned = []


    def on_click(layer, event):
        if event.type == "mouse_press":
            coords = layer.world_to_data(event.position)
            coords = tuple(int(c) for c in coords)
            lbl = layer.data[coords]

            if lbl > 0:
                print("Clicked:", lbl)
                last_clicked_label[0] = lbl
                selected_labels.add(lbl)
                update_highlight()

    segments.mouse_drag_callbacks.append(on_click)


    def update_highlight():
        
        old_visualImage[np.isin(labels, L_tuned)] = 1
        old_visualImage[np.isin(labels, M_tuned)] = 2
        old_visualImage[np.isin(labels, ML_tuned)] = 3     
        classi_layer.data = old_visualImage
            
        old_classImage[np.isin(labels, L_tuned)] = 1
        old_classImage[np.isin(labels, M_tuned)] = 2
        old_classImage[np.isin(labels, ML_tuned)] = 3     
        #classi_layer.data = old_classImage
        

        classi_layer.refresh()
        classi_layer.events.set_data()


    @view_layer.bind_key('a', overwrite=True)
    def assign_L(viewer):
        if last_clicked_label[0] is not None:
            L_tuned.append(last_clicked_label[0])
            print(f"Label {last_clicked_label[0]} → L")
            
            old_visualImage[np.isin(labels, L_tuned)] = 1
            old_visualImage[np.isin(labels, M_tuned)] = 2
            old_visualImage[np.isin(labels, ML_tuned)] = 3     
            classi_layer.data = old_visualImage
            
            classi_layer.refresh()
            classi_layer.events.set_data()
        else:
            print("Click a label first")

    @view_layer.bind_key('q', overwrite=True)
    def assign_M(viewer):
        if last_clicked_label[0] is not None:
            M_tuned.append(last_clicked_label[0])
            print(f"Label {last_clicked_label[0]} → M")
            
            old_visualImage[np.isin(labels, L_tuned)] = 1
            old_visualImage[np.isin(labels, M_tuned)] = 2
            old_visualImage[np.isin(labels, ML_tuned)] = 3     
            classi_layer.data = old_visualImage
            
            classi_layer.refresh()
            classi_layer.events.set_data()
        else:
            print("Click a label first")

    @view_layer.bind_key('j', overwrite=True)
    def assign_ML(viewer):
        if last_clicked_label[0] is not None:
            ML_tuned.append(last_clicked_label[0])
            print(f"Label {last_clicked_label[0]} → ML")
            
            old_visualImage[np.isin(labels, L_tuned)] = 1
            old_visualImage[np.isin(labels, M_tuned)] = 2
            old_visualImage[np.isin(labels, ML_tuned)] = 3     
            classi_layer.data = old_visualImage
            
            classi_layer.refresh()
            classi_layer.events.set_data()
        else:
            print("Click a label first")
            


    napari.run()
    
    Tuned_classImg_vis = old_visualImage.copy() 
    Tuned_classImg =  old_classImage.copy()  

    Tuned_classImg[np.isin(Tuned_classImg, L_tuned)] = 1
    Tuned_classImg[np.isin(Tuned_classImg, M_tuned)] = 2
    Tuned_classImg[np.isin(Tuned_classImg, ML_tuned)] = 3

    Tuned_classImg_vis[np.isin(Tuned_classImg_vis, L_tuned)] = 1
    Tuned_classImg_vis[np.isin(Tuned_classImg_vis, M_tuned)] = 2
    Tuned_classImg_vis[np.isin(Tuned_classImg_vis, ML_tuned)] = 3


    return Tuned_classImg, Tuned_classImg_vis


def Classi4RPE(tau, image):
    H, xEdge, yEdge, binNumber = getTauIntensityHistogram(tau, image)
        
    tau_thresh = (np.percentile(tau[tau>0], 15)) +12

    #get mask for M and L 

    lipImage = getTauRangeImage(image,tau, tauRange=[tau_thresh+1,tau.max()])

    lipBinary = getMask(lipImage)

    melImage = getTauRangeImage(image,tau, tauRange=[0, tau_thresh])

    melBinary = getMask(melImage)
    
    # segmentation of granules (long lifetimes and shortlifetimes)

    # By seeded water shedding
    # expansion was optimized based on the tested data sets (to reasonably matching the exact size of the granules)

    melLabel = seeded_water_shed(melImage*melBinary, min_distance = 7, expansion=2)
    lipLabel = seeded_water_shed(lipImage*lipBinary, min_distance = 3, expansion = 1)



    max_M = melLabel.max()
    lipLabel_shifted = np.where(lipLabel > 0, lipLabel + max_M, 0)  
    Overview_map = np.maximum(melLabel, lipLabel_shifted)

    # segmented image of the all segments without expansion

    melLabel_noex = seeded_water_shed(melImage*melBinary, min_distance = 7, expansion=0)
    lipLabel_noex = seeded_water_shed(lipImage*lipBinary, min_distance = 3, expansion = 1)

    max_M_noex = melLabel_noex.max()
    lipLabel_shifted_noex = np.where(lipLabel_noex > 0, lipLabel_noex + max_M_noex, 0)  
    segments_noex = np.maximum(melLabel_noex, lipLabel_shifted_noex)
    borders_noex = (find_boundaries(segments_noex, mode='outer'))
    
    # Get intensity/tau profiles of the short lifetime granules (Mlanolipofuscins / Melanin)
    #Apply tau fitting for each granule(segment) and plot the profile

    radius, intFit, tauFit = getProfiles(image, tau,melLabel, nPoly=2)
    
    # Plot values of tau & intensity ratios for each segment
    #between edge to center

    maxInt = np.max(intFit,axis=1)
    maxTau = np.max(tauFit,axis=1)
    ratioInt = intFit[:,0]/maxInt
    ratioTau = tauFit[:,-1]/tauFit[:,0]


    #Set a threshold for distinguishing Melanin from Melanolipofuscins
    # this threshold has been optimized based on the tested data sets
    thrTauRatio = 1.15

    # according the criteria classify ML clusters from M
    melFitClass = separateMLfromM(tauFit, thrTauRatio=thrTauRatio)
    
    
    #Combine L & ML classification
    # & Plotting

    melFitImage = myClassToImage(melFitClass,melLabel_noex)
    lipFitImage = (lipLabel>0)*GrainId.name['L']

    # add L M and ML together in one image
    allFitImage = np.max(np.array([melFitImage, lipFitImage]), axis=0)
    
    # create a visual image based on the 'not' expand segmentation, for better visualization
    # borders are added to see the segments clearly 
    #visual_classImage = allFitImage + (borders_noex * 4)
    visual_classImage = allFitImage.copy()
    visual_classImage[borders_noex] = 4   

    return allFitImage, visual_classImage, Overview_map

def select_data_toshow(options_data):

    selection = {"value": None, "index": None}

    def on_ok():
        sel = listbox.curselection()
        if sel:
            idx = sel[0]               
            selection["index"] = idx 
            selection["value"] = listbox.get(idx) 
        window.destroy()

    # Popup window
    window = tk.Toplevel()
    window.title("Select data to visualize")
    window.geometry("500x500")

    tk.Label(window, text="Select the data to be visualized and tuned:", pady=10).pack()

    frame = tk.Frame(window)
    frame.pack(expand=True, fill="both")

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(frame, selectmode="single")
    listbox.pack(expand=True, fill="both")

    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)

    # Insert list items
    for item in options_data:
        listbox.insert(tk.END, item)

    tk.Button(window, text="Show", command=on_ok).pack(pady=10)

    window.grab_set()
    window.wait_window()

    return selection["value"], selection["index"]

def read_ascii(path):
    with open(path, "r") as f:
        f_d = ([line.strip() for line in f if line.strip()])
        f_asc = np.array([np.fromstring(x.strip("[]"), sep=' ') for x in f_d], dtype=float)[::-1,:]
        return f_asc

def read_tif(path):
    return tifffile.imread(path)

def read_sdt(my_sdt):
    data_sdt = SdtFile(my_sdt)
    sdt = np.array(data_sdt.data)
    return sdt

def from_asci_to_tau(channel_arrays):
    tau_asc_v = (channel_arrays[0] * channel_arrays[3]) + (channel_arrays[1] * channel_arrays[4]) + (channel_arrays[2] * channel_arrays[5])
    alpha = channel_arrays[0] + channel_arrays[1] + channel_arrays[2]
    tau_asc = np.divide(tau_asc_v, alpha, out = np.zeros_like(tau_asc_v, dtype=float), where=alpha!=0)
    return tau_asc



def tune_class_click_cyclic_colored(labels, old_classImage, old_visualImage, classi_layer, view_layer):
    import napari
    # Add segment labels layer
    segments = view_layer.add_labels(labels, name="segments")

    selected_labels = set()
    last_clicked_label = [None]

    # Lists for each class
    L_tuned, M_tuned, ML_tuned, non = [], [], [], []


    # Cyclic class order
    class_order = ['L', 'ML', 'M', 'NON']
    click_counter = [0]

    
    class_colors = {
        'L': "yellow",
        'M': "green",
        'ML': "blue",
        'NON': 'white'
    }

    
    class_dict = {'L': L_tuned,  'M': M_tuned, 'ML': ML_tuned, 'NON' : non}  

    
    label_color_mapping = {}

    # Use a persistent overlay array
    overlay = old_visualImage.copy()
    
    Tuned_classImg_vis = old_visualImage.copy() 
    Tuned_classImg =  old_classImage.copy()



    def on_click(layer, event):
        if event.type == "mouse_press":
            coords = layer.world_to_data(event.position)
            coords = tuple(int(c) for c in coords)
            lbl = layer.data[coords]

            if lbl > 0:
                last_clicked_label[0] = lbl

                # Determine class cyclically
                cls = class_order[click_counter[0] % len(class_order)]
                click_counter[0] += 1

                # Add label to corresponding class list
                class_dict[cls].append(lbl)
                selected_labels.add(lbl)
                print(f"Label {lbl} → {cls}")

                # Assign color string for this label
                label_color_mapping[lbl] = class_colors[cls]

                # Update overlay layer immediately
                update_overlay(lbl, cls)

                

    segments.mouse_drag_callbacks.append(on_click)


    def update_overlay(lbl, cls):
        # Update only the clicked label in overlay
        if class_colors[cls] == "yellow":
            overlay[labels == lbl] = 1
            old_classImage[labels == lbl] = 1
            Tuned_classImg_vis[np.isin(labels, L_tuned)] = 1
            Tuned_classImg[np.isin(labels, L_tuned)] = 1
        elif class_colors[cls] == "green":
            overlay[labels == lbl] = 2
            old_classImage[labels == lbl] = 2
            Tuned_classImg_vis[np.isin(labels, M_tuned)] = 2
            Tuned_classImg[np.isin(labels, M_tuned)] = 2
        elif class_colors[cls] == "blue":
            overlay[labels == lbl] = 3
            old_classImage[labels == lbl] = 3
            Tuned_classImg_vis[np.isin(labels, ML_tuned)] = 3
            Tuned_classImg[np.isin(labels, ML_tuned)] = 3
        elif class_colors[cls] == "white":
            overlay[labels == lbl] = 4
            old_classImage[labels == lbl] = 4
            Tuned_classImg_vis[np.isin(labels, ML_tuned)] = 4
            Tuned_classImg[np.isin(labels, ML_tuned)] = 4

        classi_layer.data = overlay.copy()



        color_dict = {0: "transparent"}
        for label_id, color_str in label_color_mapping.items():
            color_dict[label_id] = color_str

        classi_layer.color = color_dict


        classi_layer.refresh()
        classi_layer._set_view_slice()

    napari.run()
   
    return Tuned_classImg, Tuned_classImg_vis


def select_data_toshow2(options_data, ints_maps, tau2_maps_arr, Classified_visImgs, ClassifiedImgs, Segments):
    
    def nap_vis(idx, ints_maps, tau2_maps_arr, Classified_visImgs, ClassifiedImgs, Segments):
        
        
        viewer = napari.Viewer()
        viewer.add_image(ints_maps[idx, :, :], name='intensity')
        intTauImage = getTauIntensityImage(ints_maps[idx, :, :], tau2_maps_arr[idx, :, :])
        viewer.add_image(intTauImage, rgb=True)
        
        ClassiLayer = viewer.add_labels(Classified_visImgs[idx, :, :], colormap=GrainId.colorNapari,
                                       name='Classi')
        

        ClassifiedImgs[idx, :, :], Classified_visImgs[idx, :, :] = tune_class_click_cyclic_colored(Segments[idx, :, :], 
                                                                    ClassifiedImgs[idx, :, :], Classified_visImgs[idx, :, :], 
                                                                    ClassiLayer, viewer)

    def on_show():

        sel = listbox.curselection()
        if not sel:
            return

        idx = sel[0]

        nap_vis(idx, ints_maps, tau2_maps_arr, Classified_visImgs, ClassifiedImgs, Segments)
        print("You have selected data:", idx)

        
    def on_close():
        window.destroy()

    # Popup window
    window = tk.Toplevel()
    window.title("Visualize & tune the classification")
    window.geometry("500x500")

    tk.Label(
        window,
        text="Select the data to be visualized:", font=("Calibri", 14), pady=10).pack()

    frame = tk.Frame(window)
    frame.pack(expand=True, fill="both")

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(frame, selectmode="single")
    listbox.pack(expand=True, fill="both")

    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)

    for item in options_data:
        listbox.insert(tk.END, item)

    btn_frame = tk.Frame(window)
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="Show", command=on_show).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Close", command=on_close).pack(side="left", padx=5)
    
    return ClassifiedImgs, Classified_visImgs
