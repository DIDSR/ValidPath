"""
---------------------------------------------------------------------------
Created on Fri Feb  4 11:42:52 2023

----------------------------------------------------------------------------

Title:        Whole Slide Image Processing Toolbox - reader module

Description:  This is the reader module for the ValidPath toolbox. It is includes ReadWsi class and several methods
              
Classes:      WSIReader
              

Methods:      wsi_reader, extract_region, extract_bounds, wsi_xml_list

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

class WSIReader:
    def __init__(self):
        pass  
    """
        This Class process the WSIs.
        
      
        :Methods:
            wsi_reader
            extract_region
            extract_bounds
            wsi_xml_list
        """
  
    def extract_region(wsi_obj,location,level,size):
        """
        This method process the WSIs and extract regions.
        
      
        :Parameters:
        wsi_obj : obj
            recieve the WSI object
            
        :Returns:
        IMG : Image
            Image data
        """
        # level = The number of levels in the slide.
        #Levels are numbered from 0 (highest resolution) to level_count - 1 (lowest resolution).
        
        return wsi_obj.read_region(location,level,size).convert('RGB')
    
    def extract_bounds(wsi_obj,bounds,level):
        """
        This method process the WSIs and extract image.
        
      
        :Parameters:
        wsi_obj : obj
            recieve the WSI object
        bounds : tuple
            recieve the locations for extracting image from WSI
        level : int
            WSI level to extract image from
            
        :Returns:
        IMG : Image
            Image data
        """

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



            
            
 