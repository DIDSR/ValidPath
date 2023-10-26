#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os, sys
import shutil
import numpy as np
from skimage import io, color  
import cv2
import os


OPENSLIDE_PATH = r'D:\openslide-win64-20171122\bin'            # OPENSLIDE BIN PATH
os.add_dll_directory(OPENSLIDE_PATH)

TEST_PATH = os.path.abspath(os.path.dirname(r'C:\Users\data'))   #root directory path
PRJ_PATH = os.path.dirname(TEST_PATH)
sys.path.insert(0, PRJ_PATH)

from tissueloc.load_slide import select_slide_level
from tissueloc.load_slide import load_slide_img
from tissueloc.locate_tissue import thresh_slide, fill_tissue_holes
from tissueloc.locate_tissue import remove_small_tissue, find_tissue_cnts
from tissueloc.locate_tissue import locate_tissue_cnts


def test_locate_tissue_seperately():
    slide_path = r"C:\Users\data\JP2.svs"                                      #SVS FILE PATH
    output_dir = os.path.join(TEST_PATH, r"C:\Users\data\20181218042458")      #OUTPUT PATH
    max_size = 10240
    # Step 1: Select the proper level
    s_level, d_factor = select_slide_level(slide_path, max_size)
    print(s_level, d_factor)
    #     # Step 2: Load Slide image with selected level
    slide_img = load_slide_img(slide_path, s_level)
    print(slide_img.shape)
    io.imsave(os.path.join(output_dir, "ori.png"), slide_img)
#     # Step 3: Convert color image to gray
    gray_img = color.rgb2gray(slide_img)
    io.imsave(os.path.join(output_dir,"gray.png"), gray_img.astype(np.uint8))
    thresh_val = 0.8
    bw_img = thresh_slide(gray_img, thresh_val)
    io.imsave(os.path.join(output_dir,"bw.png"), (bw_img*255.0).astype(np.uint8))
     # Step 5: Fill tissue holes
    bw_fill = fill_tissue_holes(bw_img)
    io.imsave(os.path.join(output_dir,"fill.png"), (bw_fill*255.0).astype(np.uint8))
#     # Step 6: Remove small tissues
    min_size = 10240
    bw_remove = remove_small_tissue(bw_fill, min_size)
    io.imsave(os.path.join(output_dir,"remove.png"), (bw_remove*255.0).astype(np.uint8))
#     # Step 7: Locate tissue regions
    cnts = find_tissue_cnts(bw_remove)
    slide_img = np.ascontiguousarray(slide_img, dtype=np.uint8)
    cv2.drawContours(slide_img, cnts, -1, (0, 255, 0), 9)
    io.imsave(os.path.join(output_dir,'cnt.png'), slide_img)
    
test_locate_tissue_seperately()


# In[3]:


def test_locate_tissue():
    slide_path = r"C:\Users\data\JP2.svs"              # svs path
    # locate tissue contours with default parameters
    cnts, d_factor = locate_tissue_cnts(slide_path,
                                        max_img_size=16384,
                                        smooth_sigma=13,
                                        thresh_val=0.80,
                                        min_tissue_size=100000)
    print("Downsampling fator is: {}".format(d_factor))
    print("There are {} contours in the slide.".format(len(cnts)))
    print(cnts[0].shape)
    print(cnts[0][:5])
    
    
test_locate_tissue()

