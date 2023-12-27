"""
---------------------------------------------------------------------------
Created on Fri Feb  4 11:42:52 2023

----------------------------------------------------------------------------

**Title:**       ValidPath Toolbox - Annotation File Generation Module

**Description:**  This is the Annotation File Generator module for the ValidPath toolbox. It is includes Annotation_Generator class and several methods
              
**Classes:**      Annotation_Generator
              

**Methods:**     There are three methods in the Annotation File Generation module as follows:

                    •	ROI_Generator.generate_map_file(input_DIR: str, output_DIR: str, file_Name: str)
                    
                    •	ROI_Generator.create_xml(input_DIR,file_Name,path_size,ROI_output_DIR,tag_name)
                    
                    •	make_region(self, x , y , id , txt,path_size,Regions)

---------------------------------------------------------------------------
Author: SeyedM.MousaviKahaki (seyed.kahaki@fda.hhs.gov)
Version ='1.0'
---------------------------------------------------------------------------
"""


import pandas as pd
import os
from lxml import etree as et
from os import walk

class Annotation_Generator:
    def __init__(self):
        pass
        
    def make_region(self, x , y , id , txt,path_size,Regions):
        """
        This method generate the XMl file structure and fill the content based on the Aperio ImageScope standard

        :Parameters:
            x : integer
                Output Directory to save the extracted annotations
            x : integer
                List of included WSIs
            txt : string
                List of XML files associated with included WSIs
            path_size : integer
                Patch size
            Regions : object
                the corresponsing XML region object
            

        :Returns:
            XML strycture
        """
        print(x)
        print(path_size)
        Region  = et.SubElement(Regions, 'Region')
        Region.set("Type", "1" ) # type "1" :  means it is rect
        Region.set("Id", str(id))
        Region.set("Text",str(txt))

        Vertices = et.SubElement(Region, 'Vertices')

        Vertex = et.SubElement(Vertices, 'Vertex') # top left
        Vertex.set('X', str(x))
        Vertex.set('Y', str(y))
        Vertex.set('Z', str(0))
        Vertex = et.SubElement(Vertices, 'Vertex') # top right
        Vertex.set('X', str(x+path_size))
        Vertex.set('Y', str(y))
        Vertex.set('Z', str(0))
        Vertex = et.SubElement(Vertices, 'Vertex') # bottom left 
        Vertex.set('X', str(x+path_size))
        Vertex.set('Y', str(y+path_size))
        Vertex.set('Z', str(0))
        Vertex = et.SubElement(Vertices, 'Vertex') # bottom right
        Vertex.set('X', str(x))
        Vertex.set('Y', str(y+path_size))
        Vertex.set('Z', str(0))  

    def create_xml(self,input_DIR, file_path  ,path_size , save_xml_path):
        """
        This method reads the map file generated uisng the ROI_Generator.generate_map_file and generated the XML annotation file based on Aperio ImageScope standard.

        :Parameters:
            input_DIR : string
                the path to the input directory of mapping file

            file_path : string
                map file name (csv)

            path_size : integer
                Size of image patch
                
            save_xml_path: string
                output directory
            

        :Returns:
            XML – the XML files
        """

        csv_file = pd.read_csv(input_DIR+file_path,index_col='WSI')

        csv_file.sort_values(by=['WSI'], inplace=True)

        #print("\n\n ****sorted***\n")
        lst_of_ann = csv_file["N_ANN"]
        #print(lst_of_ann)
        #print(len(lst_of_ann))
        count = 0

        df2 =csv_file.index
        df2 = df2.drop_duplicates()
        wsi_names = df2.values


        for wsi_name in wsi_names:

            root = et.Element('Annotations')   
            object_elem = et.SubElement(root, 'Annotation')
            object_elem.set("Name", str(lst_of_ann[count]) )

            #print(str(lst_of_ann[count]))

            Regions = et.SubElement(object_elem, 'Regions')
            


            dataf = csv_file.loc[wsi_name]
            print(wsi_name)
            if dataf.X.ndim==0 :
                count += 1
                #print(dataf['X'],"\t" ,dataf['Y'],"\t",dataf['TEXT'])
                self.make_region(dataf['X'], dataf['Y'], 1,dataf['TEXT'],path_size, Regions)
            else :  
                for k in range(len(dataf)):
                    count += 1
                    #print(dataf['X'],"\t" ,dataf['Y'],"\t",dataf['TEXT'])
                    self.make_region(dataf['X'][k], dataf['Y'][k], k+1,dataf['TEXT'][k],path_size,Regions)


            out = et.tostring(root, pretty_print=True, encoding='utf8') 
            savepath = os.path.join(save_xml_path, f'{wsi_name}.xml')
            with open(savepath, 'wb') as fd:
                fd.write(out)
                
                
    def generate_map_file(self, input_DIR,output_DIR, file_Name,tag_name):            
        """
        This method extracts different types for annotations from Whole Slide Images.
        It can save the extracted annottions to the output directory as defined in inputs.
        This code also handles several annotations per slide. 
        The output directory will be generated based on the structure of the input directories.

        :Parameters:
            input_DIR : string
                the path to the input directory of image patches

            output_DIR : str
                the path to the output directory to save the map file

            file_Name : string
                map file name (csv)
                
            tag_name : string   
                Tag name

        :Returns:
            CSV – the map file
        """

        
        print(input_DIR)
        f = []
        for (dirpath, dirnames, filenames) in walk(input_DIR):
            #print("Processing "+filenames)
            f.extend(filenames)
            break

        Data = []
        for files in f:
            split = files.split("_")
            WSI = split[1]
            TEXT = split[0]+"_"+split[1]
            X = split[3]
            Y = split[5]
            N_ANN = tag_name
            
            Data.append(
                    {
                        'WSI': WSI,
                        'TEXT': TEXT,
                        'X': X,
                        'Y': Y,
                        'N_ANN': N_ANN,
                        
                    })



        Datadf = pd.DataFrame(Data)    
        Datadf.to_csv(output_DIR+file_Name,encoding='utf-8')