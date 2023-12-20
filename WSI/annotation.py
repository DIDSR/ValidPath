"""
---------------------------------------------------------------------------
Created on Fri Feb  4 11:42:52 2023

----------------------------------------------------------------------------

Title:        ValidPath Toolbox - Annotation Module

Description:  This is the AnnotationExtractor module for the whole slide image processing toolbox. It is includes AnnotationExtractor class and several methods.
              
Classes:      AnnotationExtractor
              

Methods:      extract_ann, make_folder

---------------------------------------------------------------------------
Author: SeyedM.MousaviKahaki (seyed.kahaki@fda.hhs.gov)
Version ='3.0'
---------------------------------------------------------------------------
"""

import cv2
import numpy as np
import os
import openslide
import lxml.etree as ET
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imsave, imread
from PIL import Image
import math

highsize = False

class AnnotationExtractor :
    def __init__(self):
        pass

    
    def make_folder(self,directory):
        """
        This method creates a directory if not exist.

        :Parameters:
            directory : str
                Directory to be created.

        :Returns:
            None : None
                None.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def extract_ann(self,save_dir,XMLs,WSIs,vis=False,save_mask=False):
        """
        This method extracts different types for annotations from Whole Slide Images.
        It can save the extracted annottions to the output directory as defined in inputs.
        This code also handles several annotations per slide. 
        The output directory will be generated based on the strucutr of the input directories.

        :Parameters:
            save_dir : str
                Output Directory to save the extracted annotations

            WSIs : list
                List of included WSIs

            XMLs : list
                List of XML files associated with included WSIs

        :Returns:
            None : None
                None.
        """
        plt.rcParams.update({'font.size': 6})

        # go over all WSI
        for idx, XML in enumerate(XMLs):
            annotationID=0
            tree = ET.parse(XML)
            root = tree.getroot()
            annot = root.findall("./Annotation")
            
            for annotation in annot:
                
                final_x=[]
                final_y=[] 
                annotationID=annotation.attrib.get('Id') #annotationID+1
                Regions = root.findall("./Annotation[@Id='" + str(annotationID) + "']/Regions/Region")
                print("there are ",len(Regions) , "Regions in this annotation")
                bounds = []
                masks = []  
                lbl = annotation.attrib['Name']
                region_number=0
                print('opening: a region of ' + WSIs[idx])
                pas_img = openslide.OpenSlide(WSIs[idx])
                sw= True
                for num , Region in enumerate(Regions):
                    
                    if vis==True and save_mask ==False and sw==True  :
                        #fig,axes = plt.subplots(nrows = 1, ncols = len(Regions))
                        #fig.tight_layout(pad=5.0)
                        sw = False    
                    if vis==True and save_mask ==True and sw==True  :
                        #fig,axes = plt.subplots(nrows = len(Regions), ncols = 2)
                        #fig.tight_layout(pad=5.0)
                        sw = False
                        
                    print("\n Region numer ",num)
                    Vertices = Region.findall("./Vertices/Vertex")
                    x = []
                    y = []
                    for Vertex in Vertices:
                        x.append(int(np.float32(Vertex.attrib['X'])))
                        y.append(int(np.float32(Vertex.attrib['Y'])))
                      
                    x1=x  
                    y1=y  
                    if highsize== True:
                        x = [ math.floor(number/4) for number in x]
                        y = [math.floor(number/4) for number in y]

                    final_x.append(max(x) - min(x))
                    final_y.append(max(y) - min(y))

                    bounds.append([min(x1), min(y1)])

                    points = np.stack([np.asarray(x), np.asarray(y)], axis=1)
                    points[:,1] = np.int32(np.round(points[:,1] - min(y) )) 
                    points[:,0] = np.int32(np.round(points[:,0] - min(x) ))
                    mask = np.zeros([final_y[region_number], final_x[region_number]], dtype=np.int8)
                    
                    if(len(x)>3): #polygon
                        print("....poly")
                        cv2.fillPoly(mask, [points], color=(255,255,255))

                    if(len(x)==2): # cirlce or ellipse

                        x_c_int = int(points[1,0]/2)
                        y_c_int =int(points[1,1]/2)
                        center_coordinates=(x_c_int,y_c_int)

                        if(points[:,0][1]==points[:,1][1]): #cirlce
                            print("....circle ")
                            cv2.circle(img=mask, center =center_coordinates, radius=x_c_int , color =(255,255,255), thickness=-1)

                        else: # ellipse
                            print("....ellipse ")

                            axesLength = (x_c_int, y_c_int)
                            angle =0
                            startAngle = 0
                            endAngle = 360
                            image = cv2.ellipse(mask, center_coordinates, axesLength, angle, startAngle, endAngle, (255,255,255), thickness=-1)
                   

                    basename = os.path.basename(XML)
                    basename = os.path.splitext(basename)[0]
                    subdirm = '{}/{}/'.format(save_dir,basename)
                    
                    if save_mask == True :
                        cv2.imwrite(subdirm+basename+"_anno_"+str(annotationID)+"_reg_"+str(region_number+1)+"_mask_"+lbl+".jpg",mask)
                        print('saved <mask>')
                        print(subdirm+basename+"_anno_"+str(annotationID)+"_reg_"+str(region_number+1)+"_mask_"+lbl+".jpg")
                        if vis==True :
                            #fig = plt.figure(figsize=(2, 2))
                            plt.imshow(mask)
                            plt.show()
                            #axes[num,1].imshow(mask)
                            #axes[num,1].set_title('Mask number : {}'.format(num))
                        
                    masks.append(mask)
         
                    mask = masks[region_number]
                    if highsize== True:
                        PAS = pas_img.read_region((int(bounds[region_number][0]),  
                                                   int(bounds[region_number][1])), 
                                                  1,(final_x[region_number],final_y[region_number]))
                        # plt.imshow(PAS)
                    else:
                        PAS = pas_img.read_region((int(bounds[region_number][0]),
                                                   int(bounds[region_number][1])),
                                                  0, (final_x[region_number],final_y[region_number]))
                        # plt.imshow(PAS)
                    coord = (int(bounds[region_number][0]),int(bounds[region_number][1]))
                    # PAS = pas_img.read_region((int(bounds[region_number][0]),int(bounds[region_number][1])), 0, (final_x[region_number],final_y[region_number]))
                    PAS = np.array(PAS)[:,:,0:3]
                    for channel in range(3):
                        PAS_ = PAS[:,:,channel]
                        PAS_[masks[region_number] == 0] = 255
                        PAS[:,:,channel] = PAS_
                    subdir = '{}/{}/'.format(save_dir,basename)
                    self.make_folder(subdir)
                    #imsave(subdir + basename + '_anno_' + str(annotationID) +"_reg_"+str(region_number+1)+lblmber+1)+lbl+ "_coord_" + str(coord[0]) + "_" + str(coord[1]) + '.jpg', PAS) 
                    imsave(subdir + basename + '_anno_' + str(annotationID) +"_reg_"+str(region_number+1)+lbl+ "_coord_" + str(coord[0]) + "_" + str(coord[1]) + '.jpg', PAS) 
                    print('saved <region> of annotationID : ' + str(annotationID))
                    region_number = region_number + 1
                    
                    if vis==True and save_mask==True :
                        plt.imshow(PAS)
                        plt.show()
                        #axes[num,0].imshow(PAS)
                        #axes[num,0].set_title('Region number : {}'.format(num))
                        
                    if vis==True and save_mask ==False:
                        plt.imshow(PAS)
                        plt.show()
                        #axes[num].imshow(PAS)
                        #axes[num].set_title('Region number : {}'.format(num))
             
                #plt.show()

    