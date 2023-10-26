"""
Created on Mon Nov  6 11:23:07 2021


@author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)

Title:        WSI Patch Generatore


Description:  This code inputs should be run after the TissueRefinement code
              It will extract several patches ($Number_of_Patches=300)
              from specified images



Input:        Image: Extracted Annotation

Output:       Several patches with specific size


version ='3.0'

"""
##############################   General Imports
import random
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
import glob
import h5py
from skimage.io import imsave, imread
from skimage.transform import resize
from pathlib import Path
import sys
import pandas as pd
from scipy import misc
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as compare_ssim




# ##############################  inputs
# png_extract = parameters.PatchGenerator_png_extract
# hdf5 = False #parameters.PatchGenerator_hdf5
# open_dataset= parameters.PatchGenerator_open_dataset
# Aplha = parameters.PatchGenerator_Aplha
# INPUTDIR = parameters.PatchGenerator_INPUTDIR
# OUTPUTDIR = parameters.PatchGenerator_OUTPUTDIR

# INPUTDIR = 'Oklahoma_Extracted_New_FixedCircle_processed_Augmented_combined/'
# OUTPUTDIR = 'C:/DATA/Oklahoma_extracted_cutted_Augmented/'
# DatasetFile = 'C:/DATA/Aperio_dataset_v9.csv'

# INPUTDIR = 'AperioAnnotations_NonCancer/'
# OUTPUTDIR = 'C:/DATA/AperioAnnotations_NonCancer_Patches/'
# DatasetFile = 'C:/DATA/Aperio_dataset_v9.csv'

INPUTDIR = '2_extracted_cutted_Augmented/3DHistech/ExtractedAnnotation/'
OUTPUTDIR = 'C:/DATA/2_extracted_cutted_Augmented/3DHistech/'
DatasetFile = 'C:/DATA/Aperio_dataset_v9.csv'

hdf5_file= OUTPUTDIR+"dataset.hdf5"
root_directory = glob.glob(r'C:/DATA/'+INPUTDIR+'*')

##############################  Number of Random Patches Per Region
# Number_of_Patches = parameters.PatchGenerator_Number_of_Patches
Number_of_Patches = 36
##############################  input patch size  y==width   x==height
# input_x = parameters.PatchGenerator_input_x   
# input_y = parameters.PatchGenerator_input_y

##############################
cwd = os.getcwd()


# # Create folders if there aren't
# hdf5_dir = Path("data\\hdf5_files\\")
# hdf5_dir.mkdir(parents=True, exist_ok=True)
# png_dir = Path("data/png_files/")
# hdf5_dir.mkdir(parents=True, exist_ok=True)

png_dir = Path(OUTPUTDIR+"data/png_files/")
png_dir.mkdir(parents=True, exist_ok=True)

# path to str -We need them ahead
# png_dir=r"data/png_files/"
#hdf5_dir=r"data\\hdf5_files\\"
png_dir=OUTPUTDIR+"data/png_files/"


Dataset_ = pd.read_csv(DatasetFile)

#for on all folders
def gen_patch(INPUTDIR,input_x,input_y,Number_of_Patches,hdf5_file,DatasetFile,hdf5,png_extract,open_dataset,Aplha,OUTPUTDIR):
    """
    This function a number of pactches from extracted annotations.
    It can save the extracted annottions to the output directory as defined in inputs.
    Before running this function, please call annotation.ann_extractor.extract_ann(save_dir, XMLs, WSIs) to generate annotations. 
    The output directory will be generated based on the strucutr of the input directories.
    IF the WSI Magnification is 13X or 20X, this code will automaticall convert to 20X.
          
    :Parameters:
        root_directory : str
            Output Directory to save the extracted annotations
            
        WSIs : list
            List of included WSIs
            
        XMLs : list
            List of XML files associated with included WSIs
            
    :Returns:
        None : None
            None.
    """
    root_directory = glob.glob(r'C:/DATA/'+INPUTDIR+'*')
    for filename in root_directory:
    
        groupname = filename.split("\\")[-1]
        
        # Find Magnification
        FName = groupname.upper() + '.SVS' 
        try:
            # For Aperio
            # Magnification = Dataset_[Dataset_['Filename of initial Aperio slide'] == FName]['SVS Magnification'].item()
            # For 3DHistech
            Magnification = 13#Dataset_[Dataset_['Filename of initial 3D Histech slide'] == FName]['SVS Magnification'].item()
            if Magnification == 40:
                print(groupname+' Magnification  is 40')
            elif Magnification == 13:
                print(groupname+' Magnification  is 13')
            else:
                print('Magnification is 20')
           
            chck_group_name=True
                
           
           
            files = glob.glob( filename + r"\*.jpg")
        
            files_clean = []
            # exclude mask images
            # i=0
            for r in files:
                #print(r)
                a= r.split("\\")[-1]
                b= a.split(".")[0]
                c=b.find("mask")
                # if(c>-1):
                #     rt=files.pop()[i]\
                if(c==-1):
                    files_clean.append(r)
                # i=i+1
         
          
            # for on  all images in a folder
            for fl in files_clean:
              
            
                img = cv2.imread(fl, cv2.IMREAD_COLOR)
                if Magnification == 13:
                    print(groupname+' Rescaling 13X to 20X')
                    scale = 0.65
                    img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    
                elif Magnification == 40:
                    print(groupname+' Rescaling 40X to 20X')
                    scale = 0.5
                    img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    # img1 = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    # fig, axs = plt.subplots(1,2,figsize=(15,10))
                    # axs[0].imshow(img),axs[0].axis('off')
                    # axs[0].set_title('Original')
                    # axs[1].imshow(img1),axs[1].axis('off')
                    # axs[1].set_title('imgcv2_INTER_AREA')
                    
                    # imgcv2_INTER_CUBIC = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    # imgcv2_INTER_AREA = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    # imgcv2_INTER_LANCZOS4 = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
                    # imgcv2_INTER_NEAREST = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    # imgcv2_INTER_LINEAR = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    
                    # fig, axs = plt.subplots(2,3,figsize=(15,10))
                    # # fig.suptitle('Different Interpolations')
                    # axs[0, 0].imshow(img),axs[0,0].axis('off')
                    # axs[0, 0].set_title('Original')
                    # axs[0, 1].imshow(imgcv2_INTER_CUBIC),axs[0,1].axis('off')
                    # axs[0, 1].set_title('CUBIC')
                    # axs[0, 2].imshow(imgcv2_INTER_AREA),axs[0,2].axis('off')
                    # axs[0, 2].set_title('AREA')
                    # axs[1, 0].imshow(imgcv2_INTER_LANCZOS4),axs[1,0].axis('off')
                    # axs[1, 0].set_title('LANCZOS4')
                    # axs[1, 1].imshow(imgcv2_INTER_NEAREST),axs[1,1].axis('off')
                    # axs[1, 1].set_title('NEAREST')
                    # axs[1, 2].imshow(imgcv2_INTER_LINEAR),axs[1,2].axis('off')
                    # axs[1, 2].set_title('LINEAR')
                    
                    # cv2.PSNR(img, imgcv2_INTER_CUBIC)
                    # (score, diff) = compare_ssim(img, imgcv2_INTER_CUBIC, full=True,multichannel=True)
                    
                else:
                    print('Processing 20X')
                
                if np.mean(img) > 250:
                    continue
                
                # cv2.imshow('graycsale image',img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        
                
                rows,cols = img.shape[0], img.shape[1]
                
                b=fl.split('\\')[-1]
          
                name=b.split('.')[0]
                
             
               
                #extract patches
                for rng in range(Number_of_Patches):
             
                    end_name =name+"_patchnumber_"+str(rng) 
               
                    
                    done = True
                    breakLimit = 500
                    breakCount = 0
                    while done : 
                        print("Trying Extract: "+ name + "Break Count: " + str(breakCount))
                        breakCount = breakCount + 1
                        if breakCount > breakLimit:
                            print(">>>>>>>>>>> BREAK on "+ name)
                            break
                        coords = [(random.random()*rows, random.random()*cols)]
                        x=int(coords[0][0])
                        if(x>(rows-input_x+1)):
                            x = x-input_x+1
                   
                        y=int(coords[0][1])
                        if(y>(cols-input_x+1)):
                            y = y-input_y+1
                    
                        x_end=x+input_x
                        y_end=y+input_y
                    
                        try:
                            color_chk1 = img[x, y] > [230,230,230]#== [255,255,255]
                            color_chk2 = img[x, y_end] > [230,230,230]#== [255,255,255]
                            color_chk3 = img[x_end, y] > [230,230,230]#== [255,255,255]
                            color_chk4 = img[x_end, y_end] > [230,230,230]#== [255,255,255]
                            color_chk5 = img[round((x+x_end)/2), round((y+y_end)/2)] > [230,230,230]
                        except:
                            continue
                        if any(color_chk1) == any(color_chk2) == any(color_chk3) == any(color_chk5)== False :
                        # if any(color_chk1) == any(color_chk2) == any(color_chk3) == any(color_chk4) == any(color_chk5)== False : 
                        # if any(color_chk1) == any(color_chk2) == False : 
                            cropped_image = img[x:x_end, y:y_end]
                            
                            
                            #create png files of patches
                            if png_extract==True:
                                png_file = Path(OUTPUTDIR+"data/png_files/"+groupname+"/")
                                png_file.mkdir(parents=True, exist_ok=True)
                                print("Crearing "+png_dir +groupname+"/"+end_name+".png")
                                try:
                                    cv2.imwrite(png_dir + groupname + "/" +end_name+".png", cropped_image)
                                except:
                                    continue
                                
                              
                            #create a hdf5 file of all patches
                            if(hdf5==True):
                                if open_dataset==True:    
                                    dataset = h5py.File(hdf5_file, 'a')
                                    open_dataset=False
              
                                if chck_group_name==True:
                                    
                                    print(groupname+"_is-->>>>> new group name.")
                                    grp = dataset.create_group(groupname);
                                    chck_group_name=False
                                    
                        
                                
                                dset = grp.create_dataset(end_name, data=cropped_image)
                                print(end_name+"_is new dataset on  "+groupname+" group")
                               
                            done=False 
        except:
            print("Can not Process "+FName)
    if(hdf5==True):
        dataset.close()
	
	
	



