"""
Whole Slide Image Normalization Module of the WSI package

Several Normalization techniques are implemented in this module  

"""
# for loading/processing the images  
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.applications.vgg16 import preprocess_input 
# load json module
import json

# for everything else
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

from random import randint
import pandas as pd
import pickle
from glob import glob
from collections import Counter
import re

from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms  

def norm_HnE (img, Io=240, alpha=1, beta=0.15):
    """
    This code normalizes staining appearance of H&E stained images.
    It also separates the hematoxylin and eosing stains in to different images.

    Workflow based on the following papers:
    A method for normalizing histology slides for quantitative analysis.
    M. Macenko et al., ISBI 2009
    http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf

    Efficient nucleus detector in histopathology images. J.P. Vink et al., J Microscopy, 2013

    Original MATLAB code:
    https://github.com/mitkovetta/staining-normalization/blob/master/normalizeStaining.m

    Other useful references:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5226799/
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169875

    PROPOSED WORKFLOW:

        Input: RGB image

        Step 1: Convert RGB to OD (optical density)

        Step 2: Remove data with OD intensity less than β

        Step 3: Calculate  singular value decomposition (SVD) on the OD tuples

        Step 4: Create plane from the SVD directions corresponding to the

        two largest singular values

        Step 5: Project data onto the plane, and normalize to unit length

        Step 6: Calculate angle of each point wrt the first SVD direction

        Step 7: Find robust extremes (αth and (100−α)th 7 percentiles) of the
        angle

        Step 8: Convert extreme values back to OD space

    :Parameters:
        img : ndarray
            Input image. Can be gray-scale or in color

        Io : int, default value = 240
            Transmitted light intensity, Normalizing factor for image intensities

        alpha : int, default value = 1
            As recommend in the paper. tolerance for the pseudo-min and pseudo-max (default: 1)

        beta : float, default value = 0.15
            As recommended in the paper. OD threshold for transparent pixels (default: 0.15)

    :Returns:
        Inorm : ndarray
            Normalized Stain Vector

        H : ndarray
            H components

        E : ndarray
            E component

    """

    ######## Step 1: Convert RGB to OD ###################
    ## reference H&E OD matrix.
    #Can be updated if you know the best values for your image.
    #Otherwise use the following default values.
    #Read the above referenced papers on this topic.
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    ### reference maximum stain concentrations for H&E
    maxCRef = np.array([1.9705, 1.0308])


    # extract the height, width and num of channels of image
    h, w, c = img.shape

    # reshape image to multiple rows and 3 columns.
    #Num of rows depends on the image size (wxh)
    img = img.reshape((-1,3))

    # calculate optical density
    # OD = −log10(I)
    #OD = -np.log10(img+0.004)  #Use this when reading images with skimage
    #Adding 0.004 just to avoid log of zero.

    OD = -np.log10((img.astype(np.float)+1)/Io) #Use this for opencv imread
    #Add 1 in case any pixels in the image have a value of 0 (log 0 is indeterminate)


    ############ Step 2: Remove data with OD intensity less than β ############
    # remove transparent pixels (clear region with no tissue)
    ODhat = OD[~np.any(OD < beta, axis=1)] #Returns an array where OD values are above beta
    #Check by printing ODhat.min()

    ############# Step 3: Calculate SVD on the OD tuples ######################
    #Estimate covariance matrix of ODhat (transposed)
    # and then compute eigen values & eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))


    ######## Step 4: Create plane from the SVD directions with two largest values ######
    #project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:,1:3]) #Dot product

    ############### Step 5: Project data onto the plane, and normalize to unit length ###########
    ############## Step 6: Calculate angle of each point wrt the first SVD direction ########
    #find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:,1],That[:,0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)

    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)


    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T

    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T


    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])

    ###### Step 8: Convert extreme values back to OD space
    # recreate the normalized image using reference mixing matrix

    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # Separating H and E components

    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    return (Inorm, H, E)


def  vis_hist (source, target, matched):

            """
            To illustrate the effect of the histogram matching, this function plot for each RGB channel,
            the histogram and the cumulative histogram.

            Original code:
            https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_histogram_matching.html

            :Parameters:
                source : ndarray
                    Input image. Can be gray-scale or in color.

                target : ndarray
                    Image to match histogram of. Must have the same number of channels as image.

                channel_axis : int or None, optional
                    If None, the image is assumed to be a grayscale (single channel) image. Otherwise,
                    this parameter indicates which axis of the array corresponds to channels.

                multichannel :bool, optional
                    Apply the matching separately for each channel. This argument is deprecated: specify channel_axis instead.

            :Returns:
                None : None
                    None.

            """
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
            for i, img in enumerate((source, target, matched)):
                for c, c_color in enumerate(('red', 'green', 'blue')):
                    img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
                    axes[c, i].plot(bins, img_hist / img_hist.max())
                    img_cdf, bins = exposure.cumulative_distribution(img[..., c])
                    axes[c, i].plot(bins, img_cdf)
                    axes[c, 0].set_ylabel(c_color)

            axes[0, 0].set_title('Source')
            axes[0, 1].set_title('Reference')
            axes[0, 2].set_title('Matched')

            plt.tight_layout()
            plt.show()
            
def plot_imgs (source, target, matched):
            """
            This function plots the source image, target image, and matched image.

            :Parameters:
                source : ndarray
                    Input image. Can be gray-scale or in color.

                target : ndarray
                    Image to match histogram of. Must have the same number of channels as image.

                channel_axis : int or None, optional
                    If None, the image is assumed to be a grayscale (single channel) image. Otherwise, this parameter indicates which axis of the array corresponds to channels.

                multichannel :bool, optional
                    Apply the matching separately for each channel. This argument is deprecated: specify channel_axis instead.

            :Returns:
                None : None
                    None.

            """
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)
            for aa in (ax1, ax2, ax3):
                aa.set_axis_off()

            ax1.imshow(source)
            ax1.set_title('Source')
            ax2.imshow(target)
            ax2.set_title('Reference')
            ax3.imshow(matched)
            ax3.set_title('Matched')

            plt.tight_layout()
            plt.show()
  

                
class Normalization:
    
  
    def match_hist (source, target, visualize = True):
        
        """
        This code normalizes images.
        It manipulates the pixels of an input image so that its histogram matches the histogram of the reference image.
        If the images have multiple channels, the matching is done independently for each channel, as long as the number of channels is equal in the input image and the reference.
        Histogram matching can be used as a lightweight normalisation for image processing, such as feature matching,
        especially in circumstances where the images have been taken from different sources or in different conditions (i.e. lighting).

        Package Needed:
            skimage
        Installation:
            pip install scikit-image

        Workflow based on the following method:
            Histogram Matching, Written by Paul Bourke, January 2011
            http://paulbourke.net/miscellaneous/equalisation/

        Original code:
            https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/exposure/histogram_matching.py
            https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_histogram_matching.html



        :Parameters:
            source : ndarray
                Input image. Can be gray-scale or in color.

            target : ndarray
                Image to match histogram of. Must have the same number of channels as image.

            channel_axis : int or None, optional
                If None, the image is assumed to be a grayscale (single channel) image. Otherwise, this parameter indicates which axis of the array corresponds to channels.

            multichannel :bool, optional
                Apply the matching separately for each channel. This argument is deprecated: specify channel_axis instead.

        :Returns:
            source_matched : ndarray
                Transformed input image.

        """


        image = np.array(source, dtype=np.uint8)
        reference = np.array(target, dtype=np.uint8)


        source_matched = match_histograms(image, reference,multichannel=True)

        if visualize:
            vis_hist(source, target, source_matched)
            plot_imgs(source, target, source_matched)

        return source_matched, source, target

    
    


    def patch_norm(patch_dir,output_dir,method = 'H_E',patch_ext = 'png' ,vis = False):
        
        print("start")
        """
        This function perform patch normalization for all patches inside subfolders.

        :Parameters:
            patch_dir : str
                Input directory. This folder contains subfolders and each subfolder contains image patches.

            output_dir : str
                Output directory. Same struture as the input folders will be generated inside this directory.

            method : str, optional
                To specify the Normalization method. currently, two methods inclding 'H_E' and 'histogram_matching' are implimented

            patch_ext :str, optional
                Apply the matching separately for each channel. This argument is deprecated: specify channel_axis instead.

        :Returns:
            None : None
                None.

        """
        lst_img =[]
        for file in os.listdir(patch_dir):
            d = os.path.join(patch_dir, file)
            if os.path.isdir(d):
               
                WSI = d.rsplit('\\', 1)[1]
               

            # WSI = 'BA-10-84 HE'
            path = patch_dir+"/"+WSI+"/"
            print(path)
            # change the working directory to the path where the images are located
            os.chdir(path)

            # this list holds all the image filename
            patches = []
            saveDir = output_dir + WSI
            if not os.path.exists(saveDir):
                os.mkdir(saveDir)
            # creates a ScandirIterator aliased as files
            count = 0
            
            with os.scandir(path) as files:
              # loops through each file in the directory
            
            
                for file in files:
                    print(file.path)
                    
                    if file.name.endswith('.'+patch_ext):
                        
                        patches.append(file.name)
                        
                        img=cv2.imread(path+file.name, 1)
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        try:
                            norm_img, H_img, E_img = norm_HnE(img, Io=240, alpha=1, beta=0.15)
                            cv2.imwrite(saveDir+"/"+file.name, norm_img)
                        

                            if vis == True:
                            
                                lst_img.append([img,norm_img])
                                
                        except:
                            print(" \n \n ....... Error while executing normalization......  on ",file.path,"\n\n")
                        
                
                        
                            
        if vis == True and len(lst_img)>0:
            
            min_ = min(4,len(lst_img))
            fig,axes = plt.subplots(nrows = min_, ncols = 2,sharex = True)
            plt.figure(figsize=(10,6))
            for i in range(min_):
                    
                axes[i,0].imshow(lst_img[i][0])
                axes[i,0].set_title('\n\n orginal patch number: {}'.format(i), fontsize = 6)
                axes[i,1].imshow(lst_img[i][1])
                axes[i,1].set_title('\n\n normalized patch number: {}'.format(i), fontsize = 6)
                    
            fig.tight_layout()    
            plt.show()