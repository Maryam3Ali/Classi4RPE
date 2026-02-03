'''
Classi4RPE_GUI: is a fully simple GUI for importing data files and process them, 
allowing the user to finetune the data and change the classification. Imported data should contain: Intensity image, fitting ascii files for lifetime, and sdt data.

imported files: each data set file should contains: intensity images, ascii files of fitting, and sdt file
For finetuning: user can click on the segment layer and with each click the color can be changed (L: red, M: green, ML: blue) and will alter the classified image
Exporting: an image should be exported for the final classification + Excel sheet with calculated data for each granule in each dataset

'''

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
Ints_maps = []
d_ascii = []
No_datasets = []
dropped_folders = []

tau1_maps_o = []
tau2_maps_o = []
Classi_Images_o = []
classivis_Images_o = []
Overview_maps_o = []


tuned_class = []
tuned_vis = []

def save_ex():
    colorNapari = {1:[1,0,0],2:[0,1,0],3:[0,0,1], 4:[1,1,1]}

    colors = np.zeros((5, 4))
    colors[0] = [0, 0, 0, 1]

    for label, rgb in colorNapari.items():

        colors[label] = [rgb[0], rgb[1], rgb[2], 1.0]

    mpl_cmap = ListedColormap(colors, name="my_napari_cmap")
    
    Sdts = np.array(sdt_data)

    Sdts_sum1 = np.sum(Sdts[:, 0, :, :], axis = 3)
    Sdts_sum2 = np.sum(Sdts[:, 1, :, :], axis = 3)


    results = []
    data_no_range = np.arange(0, np.array(No_datasets), 1)
    
    Classified_visImgs = np.array(classivis_Images_o)[0, :, : :]
    ClassifiedImgs = np.array(Classi_Images_o)[0, :, : :]
    tau1_maps_arr = np.array(tau1_maps_o)[0, :, : :]
    tau2_maps_arr = np.array(tau2_maps_o)[0, :, : :]
    Segments = np.array(Overview_maps_o)
    print('all saved')
    

    for d in data_no_range:
        cal_results = {}
        plt.imsave((str(data_names[d]+"  classification"+".tiff")), Classified_visImgs[ d, :, :],cmap=mpl_cmap,
                   vmin=0, vmax=4)
        
        seg_img = Segments[0, d, :, :]

        
        labels = np.unique(seg_img)
        labels = labels[labels !=0]
        
        class_predictions = {}
        
        
        for lbl in labels:
            pixels = ClassifiedImgs[d, :, :][seg_img == lbl]
            pixels = pixels[pixels != 0]      

            if len(pixels) == 0:
                class_predictions[lbl] = 0
            else:
                class_predictions[lbl] = Counter(pixels).most_common(1)[0][0]

        predictions = np.array([class_predictions[l] for l in sorted(class_predictions)])

        
        no_pixels = (ndimage.sum(np.ones_like(seg_img), seg_img,
                                         index=np.unique(seg_img)))

        
        com = ndimage.center_of_mass(np.ones_like(seg_img),
                                 labels=seg_img,
                                 index=np.arange(1, seg_img.max()+1))
        
        
        meanT1 = ndimage.mean(tau1_maps_arr, labels=seg_img,
                         index=np.arange(1, seg_img.max()+1))
        
        meanT2 = ndimage.mean(tau2_maps_arr, labels=seg_img,
                         index=np.arange(1, seg_img.max()+1))

        
        photons_ch1 = ndimage.sum(Sdts_sum1[d, :, :], seg_img, index=labels)
        photons_ch2 = ndimage.sum(Sdts_sum2[d, :, :], seg_img, index=labels)
        
        data_ID = np.full((no_pixels.shape), str(data_names[d]))
        
        empty = np.full((no_pixels.shape), "")
        
        for i, seg in enumerate(labels):
            centers_y, centers_x = com[i]
            results.append({
                "Data ID": data_ID[i],
                "segment": seg,
                "Class [L=1, M=2, ML=3]" : predictions[i],
                "nubmer_pixels": no_pixels[i],
                "T1_mean": meanT1[i],
                "T2_mean": meanT2[i],
                "y": centers_y,
                "x": centers_x,
                "no. of photons SSC" : photons_ch1[i],
                "no. of photons LSC" : photons_ch2[i],
                "Peak_emission_wavelength": empty[i] })
        

    Export = pd.DataFrame(results)

    Export.to_excel("Results_classi4RPE.xlsx", index = False)



def classi():
    print('start -processing')
    
    tau1_maps = []
    tau2_maps = []
    Classi_Images = []
    classivis_Images = []
    Overview_maps = []
    data_no_range = np.arange(0, np.array(No_datasets), 1)
    ints_maps = np.array(Ints_maps)
   

    arr_ascii = np.array(d_ascii)
    
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
        
    Classi_Images_o.append(Classi_Images)
    classivis_Images_o.append(classivis_Images)
    Overview_maps_o.append(Overview_maps)
    tau2_maps_o.append(tau2_maps)
    tau1_maps_o.append(tau1_maps)
    

    

def vis_t():
    #print('lets see it')
               
    
    def on_show():
        tau1_maps_arr = np.array(tau1_maps_o)[0, :, :]
        tau2_maps_arr = np.array(tau2_maps_o)[0, :, :]
        ClassifiedImgs = np.array(Classi_Images_o)[0, :, :]
        Classified_visImgs = np.array(classivis_Images_o)[0, :, :]
        Segments = np.array(Overview_maps_o)[0, :, :]
        ints_maps = np.array(Ints_maps)
        
        
        sel = listbox.curselection()
        if not sel:
            return

        idx = sel[0]
            
        
        viewer = napari.Viewer()
        viewer.add_image(ints_maps[idx, :, :], name='intensity')
        intTauImage = getTauIntensityImage(ints_maps[idx, :, :], tau2_maps_arr[idx, :, :])
        viewer.add_image(intTauImage, rgb=True)
        
        ClassiLayer = viewer.add_labels(Classified_visImgs[idx, :, :], colormap=GrainId.colorNapari,
                                       name='Classi')
        

        Classi_Images_o[0][idx], classivis_Images_o[0][idx] = tune_class_click_cyclic_colored(Segments[idx, :, :], 
                                                                    ClassifiedImgs[idx, :, :], Classified_visImgs[idx, :, :], 
                                                                    ClassiLayer, viewer)

        
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

    for item in data_names:
        listbox.insert(tk.END, item)

    btn_frame = tk.Frame(window)
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="Show", command=on_show).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Close", command=on_close).pack(side="left", padx=5)
    
    


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
             Ints_maps.append(my_tif)

    my_sdt = sdt_files[0]
    sdt_d = read_sdt(my_sdt)
    sdt_data.append(sdt_d)
    
    
    
    
    


def drop(event):
    paths = root.tk.splitlist(event.data)
    folders = [Path(p) for p in paths if Path(p).is_dir()]

    if not folders:
        print("No folders detected in drop!")
        return

    for f in folders:
        if f not in dropped_folders:
            dropped_folders.append(f)
            print(f"Dropped: {f.name}")

    print(f"Total folders: {len(dropped_folders)}")


def update_listbox():
    listbox.delete(0, tk.END)
    for i, name in enumerate(data_names, start=1):
        listbox.insert(tk.END, f"{i}. {name}")
        
        
def import_all():
    if not dropped_folders:
        print("No folders to process.")
        return
    
    No_datasets.append(len(dropped_folders))
    print(f"\nNumber of folders selected: {len(dropped_folders)}\n")

    #btn_import.config(state="disabled")
    #print("\nImporting data ...\n")

    for i, folder in enumerate(dropped_folders, start=1):
        process_folder(folder, i)
        
    update_listbox()
    



root = TkinterDnD.Tk()
root.title("Classi4RPE")
root.geometry("600x800")

   
    
# for window format
classi_title = tk.Label(root, text = 'Classi4RPE', bg="lightblue", fg="black", font=("Calibri", 18, "bold"), padx=10, pady=5)
classi_title.pack()
descrp = tk.Label(root, text = 'For segmentation & Classification of Retinal Pigment Epithiluim RPE',
                  fg="blue", font=("Calibri", 14), padx=10, pady=5)
descrp.pack()



#dropping folders
label = tk.Label(
    root, text="Drag and drop folders here", font=("Calibri", 14),
    width=50, height=3)
label.pack(pady=5)

label.drop_target_register(DND_FILES)
label.dnd_bind('<<Drop>>', drop)

#first button: reading data names: just to see the list of data dropped
btn_import = tk.Button(root, text="Import data", font=("Calibri", 14), command=import_all)
btn_import.pack(pady=5)

arr_ascii = np.array(d_ascii)
ints_maps = np.array(Ints_maps)
Sdts = np.array(sdt_data)
no_datasets = np.array(No_datasets)

#second button: processing data & Apply classi4RPE
btn_process = tk.Button(root, text="classi4RPE", font=("Calibri", 14), command=lambda: classi())
btn_process.pack(pady=5)





#forth button: for visualizatoion option and finetuning
btn_vis = tk.Button(root, text="visualize & finetune", font=("Calibri", 14), command = lambda: vis_t())
btn_vis.pack(pady=5)



#fifth button: Export & save
# export the final excel sheet with all results
btn_save = tk.Button(root, text="Export results", font=("Calibri", 14), command = lambda: save_ex())
btn_save.pack(pady=5)



#closing button: close everything
btn_quit = tk.Button(root, text="close", font=("Calibri", 14), command=root.destroy)
btn_quit.pack(pady=5)

listbox = tk.Listbox(root)
listbox.pack(fill="both", expand=True, padx=10, pady=10)

root.mainloop()




