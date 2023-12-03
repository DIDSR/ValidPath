"""
#
# ---------------------------------------------------------------------------
# Created on Fri Feb  4 11:42:52 2023
#
# @author: SeyedM.MousaviKahaki (seyed.kahaki@fda.hhs.gov)
#----------------------------------------------------------------------------
# Title:        Whole Slide Image Processing Toolbox - reader module
#
# Description:  This is the reader module for the whole slide image processing 
#               toolbox. It is includes ReadWsi class and several methods
#               
#
#
# Methods:      wsi_reader
#               patch_extraction_of_tissue
#               extract_region
#               extract_bounds
#               wsi_xml_list
#               patch_extraction
#               patch_extraction_with_normalized_tiles
#
# version ='3.0'
# ---------------------------------------------------------------------------
"""


import cv2
from skimage.io import imsave, imread
from PIL import Image
import os
#os.environ['path'] ="D:\\openslide-win64-20171122\\bin"+";"+os.environ['path']
from glob import glob
import openslide
import numpy as np
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from pathlib import Path
import tifffile as tiff
import random
import matplotlib.pyplot as plt
from readwsi.TissueSegmentation import TissueSegmentation
from shapely.geometry import Polygon, Point
from readwsi.normalization import Normalization
from readwsi import normalization
import PIL

class ReadWsi:
    


            
            
            
    def extract_region(wsi_obj,location,level,size):
        
        # level = The number of levels in the slide.
        #Levels are numbered from 0 (highest resolution) to level_count - 1 (lowest resolution).
        
        return wsi_obj.read_region(location,level,size).convert('RGB')
    
    def extract_bounds(wsi_obj,bounds,level):
    

        # Specify the bounds in terms of rectangle (left, top, right, bottom)
        
        size_h = bounds[3] - bounds[1] 
        size_w = bounds[2] - bounds[0] 
        
        if(size_h<0 or size_w<0):
            print("error in bounds , try again")
            return 0
        
        size = (size_h,size_w)
        
        location = ( bounds[0] , bounds[1] )
        
        return wsi_obj.read_region(location,level,size).convert('RGB')
        
    
    def wsi_xml_list (wsis_dir):
        """
        This code process the WSIs and XML list and returns these lists.
        Only WSI are included if there is an XML file with the same name.
      
        :Parameters:
        wsis_dir : str
            Input Directory which has the original WSIs and XML files
            
        :Returns:
        WSIs : list
            List of included WSIs
            
        xml_ : list
            List of XML files associated with included WSIs
        """
        #WSIs_ = glob(wsis_dir+'/*.svs')
        files = os.listdir(wsis_dir)
        WSIs_ = [os.path.join(wsis_dir,f) for f in files]
        
        WSIs = []
        XMLs = []

        for WSI in WSIs_:
            xml_ = str.replace(WSI, 'svs', 'xml')
            xmlexist = os.path.exists(xml_)
            if xmlexist:
                print('including: ' + WSI)
                XMLs.append(xml_)
                WSIs.append(WSI)


        return (WSIs, xml_)


    def wsi_reader(WSI_path):
        """
    This code read a WSI and return the WSI object.
    This code can read the WSIs with the following formats:
    Aperio (.svs, .tif)
    Hamamatsu (.vms, .vmu, .ndpi)
    Leica (.scn)
    MIRAX (.mrxs)
    Philips (.tiff)
    Sakura (.svslide)
    Trestle (.tif)
    Ventana (.bif, .tif)
    Generic tiled TIFF (.tif)
      
    :Parameters:
        WSI_path : str
            The address to the WSI file.
            
    :Returns:
        wsi_obj : obj
            WSI object

        """

        wsi_obj = openslide.OpenSlide(WSI_path)

        return (wsi_obj)



            
            
            
class WSIpatch_extractor:
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
       
        
        tile_path = output_folder+"Imagepatches/"
        orig_tile_dir_name = output_folder+"Imagepatches/original_tiles/"
        norm_tile_dir_name = output_folder+"Imagepatches/normalized_tiles/"
        H_tile_dir_name = output_folder+"Imagepatches/H_tiles/"
        E_tile_dir_name = output_folder+"Imagepatches/E_tiles/"

        MYDIRs = [output_folder+"Imagepatches/", output_folder+"Imagepatches/original_tiles/",
                  output_folder+"Imagepatches/normalized_tiles/",output_folder+"Imagepatches/H_tiles/",
                  output_folder+"Imagepatches/E_tiles/"]
            
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
                
                
                tile_name = str(col) + "_" + str(row)
                # tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
                # print("Now processing tile with title: ", tile_name)
                temp_tile = tiles.get_tile(MaxTileLevel, (col, row))
                temp_tile_RGB = temp_tile.convert('RGB')
                temp_tile_np = np.array(temp_tile_RGB)
                # Save original tile
                
                if intensity_check:
                    intensity_cond = temp_tile_np.mean() < intensity_threshold and temp_tile_np.std() > std_threshold
                else: 
                    intensity_cond = True
                if intensity_cond:
                    print("Saving" + orig_tile_dir_name + tile_name + "_original.tif")
                    tiff.imsave(orig_tile_dir_name + tile_name + "_original.tif", temp_tile_np)
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