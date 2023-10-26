import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from datasets.wsi_dataset import Wsi_Region
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore
from keras.models import load_model
from sklearn import preprocessing
from tensorflow.python.keras.applications.resnet import ResNet50, preprocess_input
import tensorflow as tf
WEIGHTS_FOLDER = 'C:/DATA/Code/weights/'
# encoder_loaded = load_model(os.path.join(WEIGHTS_FOLDER, 'AE/epc200_im256_batch256_20220222-104322_EncoderModel.h5'),
#                                        compile=False) # epc200_im256_batch256_20220211-093252_EncoderModel.h5
encoder_loaded = load_model(os.path.join(WEIGHTS_FOLDER, 'AE/epc1024_im256_batch256_20220824-122211_EncoderModel.h5'),
                                       compile=False) # epc200_im256_batch256_20220211-093252_EncoderModel.h5

eResNet50Retrained_loaded = load_model('C:/DATA/Code/weights/ResNet50_Retrain/ResNet50_RetrainedEp10.h5',compile=False) # epc200_im256_batch256_20220211-093252_EncoderModel.h5
# Remove last prediction layer
new_model = tf.keras.models.Sequential(eResNet50Retrained_loaded.layers[:-1])
new_model.summary()


# model1 = ResNet50(weights='imagenet', pooling="avg", include_top = False) 
# model1.load_weights("C:/DATA/Code/weights/ResNet50_Retrain/best.hdf5")


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object

def compute_from_patches(wsi_object, clam_pred=None, model=None, feature_extractor=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size']
    
    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', len(roi_loader))
    mode = "w"
    for idx, (roi, coords) in enumerate(roi_loader):
        roi = roi.to(device)
        coords = coords.numpy()
        
        with torch.no_grad():
            features = feature_extractor(roi)
            # ############################################################### RESNET50 Retrained
            # broi2 = roi.numpy().reshape((roi.shape[0],256,256,3))                
            
            # features_ = new_model.predict(broi2)
            
            # # features_ = eResNet50Retrained_loaded.predict(broi2)
            # features_ = features_[:,0:1024]
            # print(features_.shape)
            
            # # from sklearn.decomposition import FactorAnalysis
            # # FA = FactorAnalysis(n_components = 1000).fit_transform(features_.transpose())
            # # print(FA.shape)
            # # if features_.shape[1] != 1024:
            # #     features_ = np.repeat(features_, 6,axis=1)
            # #     features_ = features_[:,0:1024]
            
            # # features_ = preprocessing.normalize(features_,norm='l2',axis=0)
            # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            # features_ = scaler.fit_transform(features_)
            
            # # features_ = NormalizeData(features_)
            
            # features = torch.from_numpy(features_)
            
            # # A = model(features, attention_only=True)
            # # A2_=A.numpy()
            # # print(dist(A2_,A1_))
            # ###############################################################
            
            
            # ############################################################### AE
            # broi2 = roi.numpy().reshape((roi.shape[0],256,256,3))
            # features_ = encoder_loaded.predict(broi2)
            # print(features_.shape[1])
            # # if features_.shape[1] != 1024:
            # #     features_ = np.repeat(features_, 6,axis=1)
            # #     features_ = features_[:,0:1024]
            
            # # features_ = preprocessing.normalize(features_,norm='l2',axis=0)
            # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            # features_ = scaler.fit_transform(features_)
            
            # # features_ = NormalizeData(features_)
            
            # features = torch.from_numpy(features_)
            
            # # A = model(features, attention_only=True)
            # # A2_=A.numpy()
            # # print(dist(A2_,A1_))
            # ###############################################################

            if attn_save_path is not None:
                A = model(features, attention_only=True)
           
                if A.size(0) > 1: #CLAM multi-branch attention
                    A = A[clam_pred]

                A = A.view(-1, 1).cpu().numpy()

                if ref_scores is not None:
                    for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx], ref_scores)

                asset_dict = {'attention_scores': A, 'coords': coords}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    
        if idx % math.ceil(num_batches * 0.05) == 0:
            print('procssed {} / {}'.format(idx, num_batches))

        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object