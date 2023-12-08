"""Whole Slide Image Reader"""
# for loading/processing the images  
import cv2
from skimage.io import imsave, imread
from PIL import Image

# for wsi analysis
#import openslide
#from openslide import open_slide  

# for annotation analysis
#import lxml.etree as ET
#import lxml 
 

# load json module
import json

# models 
# from keras.applications.vgg16 import VGG16 
# from keras.models import Model

# clustering and dimension reduction
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

# for everything else
import numpy as np
import os
from glob import glob
import math

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
    
    WSIs_ = glob(wsis_dir+'/*.svs')

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
    wsi_obj = openslide.OpenSlide(WSI_name)
    
    return (wsi_obj)