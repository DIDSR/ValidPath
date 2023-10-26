#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------------
# Created on Tue Nov  2 08:23:07 2021
#
# @author: SeyedM.MousaviKahaki (mousavikahaki@gmail.com)
#----------------------------------------------------------------------------
# Title:        Train Test Split
#
# Description:  This code Split the data into training and test subsets
#               by stratification over several variables (columns)
#
#
# Input:        CSV file: Dataset 
# Output:       CSV file: training and test subsets
#
# 
# Example:      train_test_split.py
#
#
# version ='3.0'
# ---------------------------------------------------------------------------
"""
Created on Mon Nov  26 17:14:38 2021

@author: SeyedM.MousaviKahaki
"""


import h5py
import cv2
import matplotlib.pyplot as plt

f = h5py.File("C:/DATA/extracted_cutted/dataset.hdf5", 'r')
f.keys()
print(f['aperio-002-0'].keys())
a=f['aperio-002-0']['aperio-002-0_anno_2_reg_7_patch number_9'][:]
plt.imshow(a)
f.close()