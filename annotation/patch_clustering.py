"""Whole Slide Image Normalization"""

"""
Created on Wed Jul  6 13:10:40 2022

@author: SeyedM.MousaviKahaki
"""

# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
# load json module
import json
# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
from glob import glob



def extract_features(file):
    """
    This function extracts image features using VGG16 model.

          
    :Parameters:
        file : str
            Input image location.
            
    :Returns:
        features : ndarray
            Extracted features using VGG16 from an image file.
            
    Examples:
        >>> from slidepro.annotation import ann_extractor
        >>> features = ann_extractor.extract_features(file)
        
    """
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


def KNNpatch_clustering(PCA_n_components = 100, PCA_random_state = 22, n_clusters = 5):
    """
    This function performs KNN clustering .
    It firstly extract features from image data using a pre-trained model (VGG16)
    Then perfomrs feature recduction using PCA metho
    Finally, it clusters the images (features) into different groups
          
    :Parameters:
        PCA_n_components : int
            PCA number of components parameter
            
        PCA_random_state : int
            PCA random state parameter
            
        n_clusters : int
            Number of final clusters
            
    :Returns:
        groups : list
            Final clustering results.
    """
    WSIs_ = glob('C:/DATA/3_extracted_cutted_Augmented_All/*')
    for W in WSIs_:
        WSI = W.rsplit('\\', 1)[1]
        print(WSI)
    
        # WSI = 'BA-10-84 HE'
        path = r"C:/DATA/3_extracted_cutted_Augmented_All/"+WSI+"/"
        # change the working directory to the path where the images are located
        os.chdir(path)
        
        # this list holds all the image filename
        flowers = []
        
        # creates a ScandirIterator aliased as files
        with os.scandir(path) as files:
          # loops through each file in the directory
            for file in files:
                if file.name.endswith('.tif'):
                  # adds only the image files to the flowers list
                    flowers.append(file.name)
                   
        data = {}
        p = r"C:/DATA/2_extracted_cutted_Augmented/Clustering/Features/flower_features_"+WSI+".pkl"
        
        # lop through each image in the dataset
        for flower in flowers:
            # try to extract the features and update the dictionary
            print('Processing '+ flower)
            try:
                feat = extract_features(flower,model)
                data[flower] = feat
            # if something fails, save the extracted features as a pickle file (optional)
            except:
                with open(p,'wb') as file:
                    pickle.dump(data,file)
              
     
        # get a list of the filenames
        filenames = np.array(list(data.keys()))
        
        # get a list of just the features
        feat = np.array(list(data.values()))
        
        # reshape so that there are 210 samples of 4096 vectors
        feat = feat.reshape(-1,4096)
        
        # # get the unique labels (from the flower_labels.csv)
        # df = pd.read_csv('flower_labels.csv')
        # label = df['label'].tolist()
        # unique_labels = list(set(label))
        
        # reduce the amount of dimensions in the feature vector
        pca = PCA(n_components=PCA_n_components, random_state=PCA_random_state)
        pca.fit(feat)
        x = pca.transform(feat)
        
        # cluster feature vectors
        # kmeans = KMeans(n_clusters=len(unique_labels),n_jobs=-1, random_state=22)
        # n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters,n_jobs=-1, random_state=22)
        kmeans.fit(x)
        
        # holds the cluster id and the images { id: [images] }
        groups = {}
        for file, cluster in zip(filenames,kmeans.labels_):
            if cluster not in groups.keys():
                groups[cluster] = []
                groups[cluster].append(file)
            else:
                groups[cluster].append(file)
        
        # Save Groups
        with open('C:/DATA/2_extracted_cutted_Augmented/Clustering/Features/groups'+WSI+'.pkl', 'wb') as handle:
            pickle.dump(groups, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # # Load Groups
        # with open('C:/DATA/2_extracted_cutted_Augmented/Clustering/Features/groups'+WSI+'.pkl', 'rb') as handle:
        #     groups = pickle.load(handle)
    
    
        ## Move Images
        import shutil
        from pathlib import Path
        for i in range (n_clusters):
            print("Processing Group: "+str(i))
            Path("Group"+str(i+1)).mkdir(parents=True, exist_ok=True)
            for j in range (len(groups[i])):
                # shutil.move("Group"+str(i)+groups[i][j],groups[i][j])
                shutil.move(groups[i][j], "Group"+str(i+1)+"/"+groups[i][j])
                
    return groups
        

# function that lets you view a cluster (based on identifier)        
def view_cluster(cluster,groups):
    plt.figure(figsize = (25,25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 50")
        files = files[:50]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
        


# view_cluster(0)

# view_cluster(1)

# view_cluster(2)

# view_cluster(3)

# view_cluster(4)

   
# # this is just incase you want to see which value for k might be the best 
# sse = []
# list_k = list(range(3, 50))

# for k in list_k:
#     km = KMeans(n_clusters=k, random_state=22, n_jobs=-1)
#     km.fit(x)
    
#     sse.append(km.inertia_)

# # Plot sse against k
# plt.figure(figsize=(6, 6))
# plt.plot(list_k, sse)
# plt.xlabel(r'Number of clusters *k*')
# plt.ylabel('Sum of squared distance');

