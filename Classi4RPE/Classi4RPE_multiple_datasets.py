#%%

import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from pathlib import Path
import tifffile
from sdtfile import SdtFile
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage
import pandas as pd
from collections import Counter

from Functions_Classi4RPE import *


data_names = []
sdt_data = []
ints_maps = []
d_ascii = []
no_datasets = []


def process_folder(folder_path: Path, folder_number: int):
    # Print folder number + folder name only
    print(f"{folder_number}. {folder_path.name}")
    name = folder_path.name
    data_names.append(name)

    # Example: count ASCII and TIFF files (optional)
    ascii_files = list(folder_path.glob("*.asc"))
    tif_files = list(folder_path.glob("*.tif")) + list(folder_path.glob("*.tiff"))
    sdt_files = list(folder_path.glob("*.sdt"))

    ascii_perdata = []
    for f in ascii_files:
         keywork_asci = ["a1[%]", "a2[%]", "a3[%]", "t1", "t2", "t3"]
         if any(k in f.name for k in keywork_asci):
             my_ascii = read_ascii(f)
             ascii_perdata.append(my_ascii)
    d_ascii.append(ascii_perdata)
             
         
    for f in tif_files:
         keywork_tif = ["Ch2-_intensity_image"]
         if any(k in f.name for k in keywork_tif):
             my_tif = read_tif(f)
             ints_maps.append(my_tif)
             

    my_sdt = sdt_files[0]
    sdt_d = read_sdt(my_sdt)
    sdt_data.append(sdt_d)
    


def drop(event):
    paths = root.tk.splitlist(event.data)
    folders = [Path(p) for p in paths if Path(p).is_dir()]

    if not folders:
        print("No folders detected in drop!")
        return
    
    no_datasets.append(len(folders))
    print(f"\nNumber of folders selected: {len(folders)}\n")

    for i, folder in enumerate(folders, start=1):
        process_folder(folder, i)

root = TkinterDnD.Tk()
root.title("Drag & Drop Folders Here")
root.geometry("500x500")

label = tk.Label(root, text="Please drag and drop folders here", width=50, height=10)
label.pack(expand=True)

# Quit button
btn_quit = tk.Button(root, text="Quit", command=root.destroy)
btn_quit.pack(pady=10)

# Register drag & drop
label.drop_target_register(DND_FILES)
label.dnd_bind('<<Drop>>', drop)

root.mainloop()

#%%

arr_ascii = np.array(d_ascii)
ints_maps = np.array(ints_maps)
Sdts = np.array(sdt_data)
no_datasets = np.array(no_datasets)

#%%

tau1_maps = []
tau2_maps = []
Classi_Images = []
classivis_Images = []
Overview_maps = []
data_no_range = np.arange(0, no_datasets, 1)

for i in data_no_range:
    
    ch1_data = [arr_ascii[i, 0, :, :], arr_ascii[i, 1, :, :],
              arr_ascii[i, 2, :, :], arr_ascii[i, 3, :, :],
              arr_ascii[i, 4, :, :], arr_ascii[i, 5, :, :]]
    
    ch2_data = [arr_ascii[i, 6, :, :], arr_ascii[i, 7, :, :],
              arr_ascii[i, 8, :, :], arr_ascii[i, 9, :, :],
              arr_ascii[i, 10, :, :], arr_ascii[i, 11, :, :]]
    
    tau1 = from_asci_to_tau(ch1_data)
    tau1 = np.nan_to_num(tau1)
    tau1_maps.append(tau1)
    tau2 = from_asci_to_tau(ch2_data)
    tau2 = np.nan_to_num(tau2)
    tau2_maps.append(tau2)
    
    classi_Image, classivis_Image, segments = Classi4RPE(tau2, ints_maps[i, :, :])
    Classi_Images.append(classi_Image)
    classivis_Images.append(classivis_Image)
    Overview_maps.append(segments)


tau1_maps_arr = np.array(tau1_maps)
tau2_maps_arr = np.array(tau2_maps)
ClassifiedImgs = np.array(Classi_Images)
Classified_visImgs = np.array(classivis_Images)
Segments = np.array(Overview_maps)




#%%

selected_show, order_data = select_data_toshow(data_names)


if selected_show is not None:
    print("You have selected data:", selected_show)

    viewer = napari.Viewer()
    viewer.add_image(ints_maps[order_data, :, :], name='intensity')
    intTauImage = getTauIntensityImage(ints_maps[order_data, :, :],tau2_maps_arr[order_data, :, :])
    viewer.add_image(intTauImage, rgb=True)
    
    ClassiLayer = viewer.add_labels(Classified_visImgs[order_data, :, :], colormap=GrainId.colorNapari,
                                   name='Classi')
    
    # Finetuning & manual modifications on the classification
    #By clicking on the sgmented image then a => L, q => M, j => ML
    ClassifiedImgs[order_data, :, :], Classified_visImgs[order_data, :, :] = tune_class_click(Segments[order_data, :, :], ClassifiedImgs[order_data, :, :], 
                            Classified_visImgs[order_data, :, :], ClassiLayer, viewer)

else:
    print("You didn't select a data to visualize it" )

#%% save & export data


colorNapari = {1:[1,0,0],2:[0,1,0],3:[0,0,1], 4:[1,1,1]}

colors = np.zeros((5, 4))
colors[0] = [0, 0, 0, 1]

for label, rgb in colorNapari.items():

    colors[label] = [rgb[0], rgb[1], rgb[2], 1.0]

mpl_cmap = ListedColormap(colors, name="my_napari_cmap")

segment_numbers = []
empty = []
data_ID = []
centers = []
mean_tau_ch2 = []
mean_tau_ch1 = []
no_photons1 = []
no_photons2 = []
classi_preds = []
pix = []

for d in data_no_range:
    plt.imsave((str(data_names[d]+"  classification"+".tiff")), Classified_visImgs[d, :, :],cmap=mpl_cmap,
               vmin=0, vmax=4)
    
    labels = np.unique(Segments[d, :, :])
    class_predictions = {}
    
    for lbl in labels:
        pixels = ClassifiedImgs[d, :, :][Segments[d, :, :] == lbl]
        pixels = pixels[pixels != 0]      

        if len(pixels) == 0:
            class_predictions[lbl] = 0
        else:
            class_predictions[lbl] = Counter(pixels).most_common(1)[0][0]

    no_pixels = (ndimage.sum(np.ones_like(Segments[d, :, :]), Segments[d, :, :],
                                     index=np.unique(Segments[d, :, :])))

    #class_predictions = {lbl: np.unique(ClassifiedImgs[d, :, :][Segments[d, :, :] == lbl])[0] for lbl in labels}
    predictions = np.array([class_predictions[l] for l in sorted(class_predictions)])

    
    for i in (np.unique(Segments[d, :, :])[1:]):
        segment = (Segments[d, :, :]==i)
        cen_of_mass = ndimage.center_of_mass(Segments[d, :, :])
        centers.append(cen_of_mass)
        Lifetau1 = tau1_maps_arr[d, :, :][segment]
        mean_tau1 = np.mean(Lifetau1)
        mean_tau_ch1.append(mean_tau1)
        Lifetau2 = tau2_maps_arr[d, :, :][segment]
        mean_tau2 = np.mean(Lifetau2)
        mean_tau_ch2.append(mean_tau2) 
        #number of photons per granule
        photons_ch1 = np.sum((np.sum(Sdts[d, 0, :, :, :], axis = 2))[segment])
        photons_ch2 = np.sum((np.sum(Sdts[d, 1, :, :, :], axis = 2))[segment])
        no_photons1.append(photons_ch1)
        no_photons2.append(photons_ch2)
        segment_numbers.append(i)
        empty.append("")
        data_ID.append(data_names[d])
        pix.append(no_pixels[i])
        classi_preds.append(predictions[i])
        #print(i)
    

coord = np.array(centers)
meanT2 = np.array(mean_tau_ch2)
meanT1 = np.array(mean_tau_ch1)
photons1 = np.array(no_photons1)
photons2 = np.array(no_photons2)
no_seg_per = np.array(segment_numbers)
empty_arr = np.array(empty)
title_data = np.array(data_ID)
classi_p_res = np.array(classi_preds)
no_pixels_arr = np.array(pix)



Export_results = {"Data ID":title_data, "segment":no_seg_per, "Class [L=1, M=2, ML=3]":classi_p_res,
                  "X":coord[:, 1], "Y":coord[:, 0],
                  "mean_tau_Ch1":meanT1, "mean_tau_Ch2":meanT2,
                  "no. of pixels":no_pixels_arr, "no. of photons SSC":photons1, "no. of photons LSC":photons2, 
                  "Peak Emmision Wavelength":empty}
Export = pd.DataFrame(Export_results)

Export.to_excel("Results_classi4RPE.xlsx", index = False)


    