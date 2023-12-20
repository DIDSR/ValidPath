"""
---------------------------------------------------------------------------
Created on Fri Feb  4 11:42:52 2023

----------------------------------------------------------------------------

Title:        ValidPath Toolbox - patch module

Description:  This is the patch module for the whole slide image processing toolbox. It is includes two classes and several methods
              
Classes:      WSIpatch_extractor, PatchExtractor
              

Methods:      patch_extraction, patch_extraction_of_tissue, patch_extraction_with_normalized_tiles, find_between, gen_patch

---------------------------------------------------------------------------
Author: SeyedM.MousaviKahaki (seyed.kahaki@fda.hhs.gov)
Version ='3.0'
---------------------------------------------------------------------------
"""

import numpy as np
import cv2
from skimage.io import imsave, imread
from PIL import Image
import os
from glob import glob
import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from pathlib import Path
import tifffile as tiff
import random
import matplotlib.pyplot as plt
#from readwsi.TissueSegmentation import TissueSegmentation
from shapely.geometry import Polygon, Point
#from readwsi.normalization import Normalization
#from readwsi import normalization
import PIL
import h5py
#import sys
import pandas as pd
#from scipy import misc
import matplotlib.pyplot as plt     
            
            
class WSIpatch_extractor:
    def __init__(self):
        pass
    def patch_extraction(wsi_obj,patch_size,output_folder,random_state,visualize,intensity_check,intensity_threshold,std_threshold,patch_number=-1):
        """
        this function  Generate object for tiles using the DeepZoomGenerator and divided 
        the svs file into tiles of size 256 with no overlap.
        then  processing and saving each tile to local directory.
        
          
        :Parameters:
        wsi_obj : obj
            WSI object.
        patch_size: int
            size tiles
        output_folder : str
            path root folder to save tiles
        perform_segmentation_state: bool
        random_state : bool
            
        """
        # Generate object for tiles using the DeepZoomGenerator
        tiles = DeepZoomGenerator(wsi_obj, tile_size= patch_size, overlap=0, limit_bounds=False)
        # Here, we have divided our svs into tiles of size 256 with no overlap.
     
        # The tiles object also contains data at many levels.
        # To check the number of levels
        print("The number of levels in the tiles object are: ", tiles.level_count)
        print("The dimensions of data in each level are: ", tiles.level_dimensions)
        # Total number of tiles in the tiles object
        print("Total number of tiles = : ", tiles.tile_count)

        ###### processing and saving each tile to local directory
        #print("<<<<<<<<<<<<<<<<")
        MaxTileLevel = len(tiles.level_tiles) - 1
        
        cols, rows = tiles.level_tiles[MaxTileLevel]
       
        #print(tiles)
        #import pdb; pdb.set_trace()
        tile_path = output_folder+"Imagepatches/"
        orig_tile_dir_name = output_folder+"Imagepatches/"
        #norm_tile_dir_name = output_folder+"Imagepatches/normalized_tiles/"
        #H_tile_dir_name = output_folder+"Imagepatches/H_tiles/"
        #E_tile_dir_name = output_folder+"Imagepatches/E_tiles/"

        # MYDIRs = [output_folder+"Imagepatches/", output_folder+"Imagepatches/original_tiles/",
                  # output_folder+"Imagepatches/normalized_tiles/",output_folder+"Imagepatches/H_tiles/",
                  # output_folder+"Imagepatches/E_tiles/"]
        MYDIRs = [output_folder+"Imagepatches/"]
            
        for dr in MYDIRs:
            CHECK_FOLDER = os.path.isdir(dr)
            # If folder doesn't exist, then create it.
            if not CHECK_FOLDER:
                os.makedirs(dr)
                print("created folder : ", dr)
            else:
                print(dr, "folder already exists.")
        
       
        ro = patch_number
        co = 2
        axes=[]
        
        #fig = plt.figure(figsize=(18, 10))
        
        counter = 0 
        flag_counter = True
        for row in range(rows):
            if flag_counter==False:
                break
            for col in range(cols):
                   
                if random_state==True :
                        
                    row = random.randint(0,rows-1)
                    col = random.randint(0,cols-1)   
       # sw = False
       # for row in range(rows):
         #   if sw == True:
        #        break
        #    for col in range(cols):
                
                
                #tile_name = str(col) + "_" + str(row)
                # tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
                # print("Now processing tile with title: ", tile_name)
                temp_tile = tiles.get_tile(MaxTileLevel, (col, row))
                #getting the coordinates
                temp_tile_coor = tiles.get_tile_coordinates(MaxTileLevel, (col, row))
                xx = temp_tile_coor[0][0]
                yy = temp_tile_coor[0][1]
                
                
                tile_name = str(counter)+"_"+str(0) + "_x_" + str(xx) + "_y_" + str(yy) + "_a_" + "100.00"
                
                
                temp_tile_RGB = temp_tile.convert('RGB')
                temp_tile_np = np.array(temp_tile_RGB)
                # Save original tile
                
                # to check intensity mean and intensity standard deviation of the tiles (To exclude non-informative tiles)
                if intensity_check:
                    intensity_cond = temp_tile_np.mean() < intensity_threshold and temp_tile_np.std() > std_threshold
                else: 
                    intensity_cond = True
                if intensity_cond:
                    print("Saving" + orig_tile_dir_name + tile_name + ".tif")
                    tiff.imsave(orig_tile_dir_name + tile_name + ".tif", temp_tile_np)
                   # fig = plt.figure(figsize=(7, 7))
                    #fig.add_subplot(ro, co, 1)
                    #plt.imshow(temp_tile_np)
                    #plt.axis('off')
                    
                    #plt.title("patch number :" + str(counter+1))
                   # fig = plt.figure(figsize=(7, 7))
                    #plt.plot(temp_tile_np)
                    #plt.figure(i+1)
                    # plt.rcParams.update({'font.size': 8})
                    # axes.append( fig.add_subplot(ro, co, counter+1) )
                    # subplot_title=("patch number :" + str(counter+1))
                    # axes[-1].set_title(subplot_title)  
                    if visualize ==1:
                        plt.imshow(temp_tile_np)
                        plt.show()
                    
                    if patch_number >0 :
                        counter  += 1
                    
                        if patch_number == counter :
                           
                            flag_counter = False
                            break
                       # sw =True
                       # break
       # return (temp_tile_np)
        #fig.tight_layout(pad=1)
        #plt.show()
        
        
    def patch_extraction_of_tissue(slidepath,patch_size ,output_folder, number_of_patches=1 , vis = False):
        
        # vis   =   visualize state , if it is True == output will be displayed
        std_threshold = 15
        intensity_threshold = 250
        img_array = []
        
        #Minimum tissue area
        min_tissue_size =10000
        #Max height and width for the size of slide with selected level
        max_img_size =10000
        #Gaussian smoothing sigma
        smooth_sigma = 13
        #Thresholding value
        thresh_val =0.8
        
        s_level, d_factor ,slide_shape  = TissueSegmentation.test_locate_tissue_seperately(slidepath,output_folder,
                                                                                   min_tissue_size,max_img_size,smooth_sigma,thresh_val)
        cnts = TissueSegmentation.test_locate_tissue(slidepath,min_tissue_size,max_img_size,smooth_sigma,thresh_val)
        
        
        Slide = openslide.OpenSlide(slidepath)
        region = (0, 0)
        level = s_level
        factor = d_factor
        w_ = slide_shape[0]
        h_ = slide_shape[1]
        size = (slide_shape[0], slide_shape[1])


        list_of_polygons= []
        for i,cnt in enumerate(cnts) :
            lst_of_tuples = [] 
            x=[]
            y=[] 

            for j,cn in enumerate(cnt):

                x.append(cn[0][0])
                y.append(cn[0][1])
            lst_of_tuples = list(zip(x,y))
            list_of_polygons.append(lst_of_tuples)

        #for num_p in range(number_of_patches) : 
        num_p = 0
        while num_p !=  number_of_patches:    
            n = 1
            while n>0:
                rand_x = random.randint(1,w_)
                rand_y = random.randint(1,h_)
                point = Point(rand_x,rand_y)
                polygon =  Polygon(lst_of_tuples)                                # Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
                is_correct = polygon.contains(point) 
                if is_correct == True:
                    n = -1
                    break


            spointx, spointy = rand_x*factor, rand_y*factor #multipled by factor to get the original co-ordinates as in WSI and not the as per the level 


            patchimg = Slide.read_region((spointx, spointy), level, (patch_size, patch_size))                        
            #patchimg.convert('RGB')

           # cv2.imwrite(f"C:/Users/masoud/data/patch_{str(num_p)}.tif", np.array(patchimg))
            temp_tile_RGB = patchimg.convert('RGB')
            temp_tile_np = np.array(temp_tile_RGB)
            
            # to check intensity mean and intensity standard deviation of the tiles (To exclude non-informative tiles)
            if temp_tile_np.mean() < intensity_threshold and temp_tile_np.std() > std_threshold:
                tiff.imsave(output_folder + f"/patch_{str(num_p)}.tif", temp_tile_np)
                print(output_folder+ f"/patch_{str(num_p)}.tif")
                #print("-------------", type(patchimg))
                num_p += 1
        
                
                img_array.append(temp_tile_np)
       
        l = [ 4,number_of_patches]
        min_ = min(l)
        
        if vis ==True :
            fig,axes = plt.subplots(nrows = 1, ncols = min_)
      
            for i,x in enumerate(img_array) :
                if i==4 :
                    break
                
           
                axes[i].imshow(x)
                
                
            plt.show()
            
            
    def patch_extraction_with_normalized_tiles(wsi_obj,patch_size,output_folder,random_state=True ,patch_number=-1):
            """
            this function  Generate object for tiles using the DeepZoomGenerator and divided 
            the svs file into tiles of size 256 with no overlap.
            then  processing and saving each tile to local directory.


            :Parameters:
            wsi_obj : obj
                WSI object.
            patch_size: int
                size tiles
            output_folder : str
                path root folder to save tiles
            perform_segmentation_state: bool
            random_state : bool

            """
            # Generate object for tiles using the DeepZoomGenerator
            tiles = DeepZoomGenerator(wsi_obj, tile_size= patch_size, overlap=0, limit_bounds=False)
            # Here, we have divided our svs into tiles of size 256 with no overlap.

            # The tiles object also contains data at many levels.
            # To check the number of levels
            print("The number of levels in the tiles object are: ", tiles.level_count)
            print("The dimensions of data in each level are: ", tiles.level_dimensions)
            # Total number of tiles in the tiles object
            print("Total number of tiles = : ", tiles.tile_count)
            #print(">>>>>>>>>>>>>>>>")
            std_threshold = 15
            intensity_threshold = 250
            MaxTileLevel = len(tiles.level_tiles) - 1
            ###### processing and saving each tile to local directory
            cols, rows = tiles.level_tiles[MaxTileLevel]
            
            
            orig_tile_dir_name = output_folder+"Imagepatches/original_tiles/"
            norm_tile_dir_name = output_folder+"Imagepatches/normalized_tiles/"
            H_tile_dir_name = output_folder+"Imagepatches/H_tiles/"
            E_tile_dir_name = output_folder+"Imagepatches/E_tiles/"
            
            MYDIRs = [output_folder+"Imagepatches/original_tiles/",output_folder+"Imagepatches/normalized_tiles/",
                output_folder+"Imagepatches/H_tiles/",output_folder+"Imagepatches/E_tiles/"]
            
            for dr in MYDIRs:
                CHECK_FOLDER = os.path.isdir(dr)
                # If folder doesn't exist, then create it.
                if not CHECK_FOLDER:
                    os.makedirs(dr)
                    print("created folder : ", dr)
                else:
                    print(dr, "folder already exists.")
                    
            counter = 0 
            flag_counter = True
            c = 0
            ro = patch_number*4
            co = 4
            axes=[]
            fig=plt.figure()  
            plt.rcParams.update({'font.size': 8})
            for row in range(rows):
                if flag_counter==False:
                    break
                for col in range(cols):
                   
                    if random_state==True :
                        
                        row = random.randint(0,rows-1)
                        col = random.randint(0,cols-1)                        
                        
                    tile_name = str(col) + "_" + str(row)
                    # tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
                    # print("Now processing tile with title: ", tile_name)
                    
                    #print(MaxTileLevel, col, row)
                    temp_tile = tiles.get_tile(MaxTileLevel, (col, row))
                    
                    temp_tile_RGB = temp_tile.convert('RGB')
                    temp_tile_np = np.array(temp_tile_RGB)
                    
                    
                    # to check intensity mean and intensity standard deviation of the tiles (To exclude non-informative tiles)
                    if temp_tile_np.mean() < intensity_threshold and temp_tile_np.std() > std_threshold:
                       # print("Processing tile number:", tile_name)
                        norm_img, H_img, E_img = normalization.norm_HnE(temp_tile_np, Io=240, alpha=1, beta=0.15)
                        # Save original tile
                        tiff.imsave(orig_tile_dir_name + tile_name + "_original.tif", temp_tile_np)
                        # Save the norm tile, H and E tiles
                        tiff.imsave(norm_tile_dir_name + tile_name + "_norm.tif", norm_img)
                        tiff.imsave(H_tile_dir_name + tile_name + "_H.tif", H_img)
                        tiff.imsave(E_tile_dir_name + tile_name + "_E.tif", E_img)
                       
                        fig = plt.figure(figsize=(15, 15))
                        axes.append( fig.add_subplot(ro, co, c+1) )
                        subplot_title=("patch number (original_img) :" + str(counter+1))
                        axes[-1].set_title(subplot_title)  
                        plt.imshow(temp_tile_np)
                        axes.append( fig.add_subplot(ro, co, c+2) )
                        subplot_title=("patch number (norm_img) :" + str(counter+1))
                        axes[-1].set_title(subplot_title)  
                        plt.imshow(norm_img)
                        axes.append( fig.add_subplot(ro, co, c+3) )
                        subplot_title=("patch number (H_img) :" + str(counter+1))
                        axes[-1].set_title(subplot_title)  
                        plt.imshow(H_img)
                        axes.append( fig.add_subplot(ro, co, c+4) )
                        subplot_title=("patch number (E_img) :" + str(counter+1))
                        axes[-1].set_title(subplot_title)  
                        plt.imshow(E_img)
                        c = c+4
                        if patch_number >0 :
                            counter  += 1
                        
                            if patch_number == counter :
                         
                                flag_counter = False
                                break

                    #else:
                        #print("NOT PROCESSING TILE:", tile_name)
            #fig.tight_layout()
            plt.show()
            

class PatchExtractor :
    def __init__(self):
        pass
    #for on all folders
    
    def find_between(self, s, first, last ):
        try:
            start = s.index( first ) + len( first )
            end = s.index( last, start )
            return s[start:end]
        except ValueError:
            return ""
    
    def gen_patch(self, INPUTDIR,PatchSize,Number_of_Patches,intensity_check,intensity_threshold,OUTPUTDIR):
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
        
        
        std_threshold = 15
        chck_group_name=True
        open_dataset = True 
        save_hdf5 = False
        save_png = True
        input_x = PatchSize[0]
        input_y = PatchSize[1]
        hdf5_file= OUTPUTDIR+"dataset.hdf5"
        root_directory = glob(r''+INPUTDIR+'*')
        print(">>>>>>>>>")
        print(root_directory)
        png_dir = OUTPUTDIR+"Imagepatches/"
        if not os.path.exists(png_dir):
            os.makedirs(png_dir)
                
        print(png_dir)
        
        for filename in root_directory:
        
            groupname = os.path.basename(filename)
            #FName = groupname.upper()

            #files = glob( filename + r"\*.jpg")
            # adjustment for compatibility with linux (because "\" is the wrong type of slash for unix):
            files = glob(os.path.join(filename, "*.jpg"))
            #import pdb; pdb.set_trace()
            #files = [f.upper() for f in files]
        
            files_clean = []
            # exclude mask images
            # i=0
            for r in files:
                subr = self.find_between( r, "_coord_", "." )
                
                xx = subr.split("_")[1]
                yy = subr.split("_")[0]

                a= r.split("\\")[-1]
                b= a.split(".")[0]
                c=b.find("mask")
                # if(c>-1):
                #     rt=files.pop()[i]\
                if(c==-1):
                    files_clean.append(r)

            # for on  all images in a folder
            for fl in files_clean:
              
            
                img = cv2.imread(fl, cv2.IMREAD_COLOR)
                plt.imshow(img)
                plt.show()
    
                # to check intensity mean and intensity standard deviation of the tiles (To exclude non-informative tiles)
                if np.mean(img) > intensity_threshold:
                    continue

                # Don't use regions that are too small
                print(img.shape)
                min_acceptable_height, min_acceptable_width = 2*input_y, 2*input_x
                if img.shape[0] < min_acceptable_height or img.shape[1] < min_acceptable_width:
                    continue
              
                rows,cols = img.shape[0], img.shape[1]
                
                b=fl.split('\\')[-1]
          
                name=b.split('.')[0]

                #extract patches
                for rng in range(Number_of_Patches):
             
                    #end_name =name+"_patchnumber_"+str(rng) 

                    done = True
                    breakLimit = 500
                    breakCount = 0
                    while done : 
                        
                        breakCount = breakCount + 1
                        if breakCount > breakLimit:
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
                        
                        
                        # to include patches with high intensity in corner of the patches
                        try:
                            color_chk1 = img[x, y] > [intensity_threshold,intensity_threshold,intensity_threshold]
                            color_chk2 = img[x, y_end] > [intensity_threshold,intensity_threshold,intensity_threshold]
                            color_chk3 = img[x_end, y] > [intensity_threshold,intensity_threshold,intensity_threshold]
                            color_chk4 = img[x_end, y_end] > [intensity_threshold,intensity_threshold,intensity_threshold]
                            color_chk5 = img[round((x+x_end)/2), round((y+y_end)/2)] > [intensity_threshold,intensity_threshold,intensity_threshold]
                            
                        except:
                            continue
                        
                        if intensity_check:
                            intensity_cond = any(color_chk1) == any(color_chk2) == any(color_chk3) == False
                        else:
                            intensity_cond = True
                        
                        # Check three corner have high intensity
                        if intensity_cond : 
                            cropped_image = img[x:x_end, y:y_end]
                            
                            # add location of annotation (xx,yy) to patch location (x,y)
                            yFinal = int(xx) + x
                            xFinal = int(yy) + y
                            # plt.imshow(cropped_image)
                            # plt.show()
                            #create png files of patches
                            if save_png==True:
                                png_file = Path(OUTPUTDIR+"Imagepatches/"+groupname+"/")
                                png_file.mkdir(parents=True, exist_ok=True)
                                print(png_dir)
                                #end_name = str(rng)+"_"+groupname + "_x_" + str(x) + "_y_" + str(y) + "_a_" + "100.00"
                                end_name = str(rng)+"_"+groupname + "_x_" + str(xFinal) + "_y_" + str(yFinal) + "_a_" + "100.00"
                                print("Creating "+png_dir +groupname+"/"+end_name+".png")
                                try:
                                    cv2.imwrite(png_dir + groupname + "/" +end_name+".png", cropped_image)
                                except:
                                    continue
  
                            #create a hdf5 file of all patches
                            if(save_hdf5==True):
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

        if(save_hdf5==True):
            dataset.close()
	
	
	
