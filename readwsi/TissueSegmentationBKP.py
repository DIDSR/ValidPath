#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os, sys
import shutil
import numpy as np
from skimage import io, color  
import cv2
import matplotlib.pyplot as plt

import os
os.environ['path'] ="D:\\openslide-win64-20171122\\bin"+";"+os.environ['path']
import openslide

#TEST_PATH = os.path.abspath(os.path.dirname(r'C:\Users\masoud\data'))   #root directory path
#PRJ_PATH = os.path.dirname(TEST_PATH)
#sys.path.insert(0, PRJ_PATH)

from tissueloc.load_slide import select_slide_level
from tissueloc.load_slide import load_slide_img
from tissueloc.locate_tissue import thresh_slide, fill_tissue_holes
from tissueloc.locate_tissue import remove_small_tissue, find_tissue_cnts
from tissueloc.locate_tissue import locate_tissue_cnts

class TissueSegmentation:


    def test_locate_tissue_seperately(slide_path, output_dir, min_size, max_size,smooth_sigma,thresh_val , vis =False):
        
        # vis   =   visualize state , if it is True == output will be displayed  
        
        # min_size = Minimum tissue area
       
        # max_size = Max height and width for the size of slide with selected level
     
        # smooth_sigma = Gaussian smoothing sigma
     
        # thresh_val = Thresholding value
        
        
     
        # Step 1: Select the proper level
        #fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 8))
        if vis==True :
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 15), sharex=True, sharey=True)
            for aa in (ax1, ax2, ax3):
                aa.set_axis_off()
                
        s_level, d_factor = select_slide_level(slide_path, max_size)
        print("level dimensions that processed is : ",s_level, "   downsample factors is :",d_factor)
        
        #     # Step 2: Load Slide image with selected level
        slide_img = load_slide_img(slide_path, s_level)
        print("shape of image that processed : ",slide_img.shape)
        io.imsave(os.path.join(output_dir, "ori.png"), slide_img)

            
    #     # Step 3: Convert color image to gray
        gray_img = color.rgb2gray(slide_img)
        io.imsave(os.path.join(output_dir,"gray.png"), gray_img.astype(np.uint8))
        thresh_val = 0.8
        bw_img = thresh_slide(gray_img, thresh_val)
        io.imsave(os.path.join(output_dir,"bw.png"), (bw_img*255.0).astype(np.uint8))
        
         # Step 5: Fill tissue holes
        bw_fill = fill_tissue_holes(bw_img)
        fill_ = (bw_fill*255.0).astype(np.uint8)
        io.imsave(os.path.join(output_dir,"fill.png"), fill_)
    #     # Step 6: Remove small tissues
    
   
        
        bw_remove = remove_small_tissue(bw_fill, min_size)
        io.imsave(os.path.join(output_dir,"remove.png"), (bw_remove*255.0).astype(np.uint8))
        
    #     # Step 7: Locate tissue regions
        if vis==True :
            ax1.imshow(slide_img)
            ax1.set_title('slide_img')
            ax2.imshow(fill_)
            ax2.set_title('fill')

        
        cnts = find_tissue_cnts(bw_remove)
        slide_img = np.ascontiguousarray(slide_img, dtype=np.uint8)
        cv2.drawContours(slide_img, cnts, -1, (0, 255, 0), 9)
        io.imsave(os.path.join(output_dir,'cnt.png'), slide_img)
        
        if vis==True :
            ax3.imshow(slide_img)
            ax3.set_title('segmented')
            plt.tight_layout()
            plt.show()

             
        return s_level, d_factor,slide_img.shape 
    
    
    
    def test_locate_tissue(slide_path, min_tissue_size, max_img_size,smooth_sigma,thresh_val):
        
         # min_tissue_size = Minimum tissue area
       
        # max_img_size = Max height and width for the size of slide with selected level
     
        # smooth_sigma = Gaussian smoothing sigma
     
        # thresh_val = Thresholding value
        
        
        # locate tissue contours with default parameters
        cnts, d_factor = locate_tissue_cnts(slide_path,
                                            max_img_size,
                                            smooth_sigma,
                                            thresh_val,
                                            min_tissue_size)
        print("Downsampling fator is: {}".format(d_factor))
        print("There are {} contours in the slide.".format(len(cnts)))
     
        return cnts
        









