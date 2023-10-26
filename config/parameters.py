# generate_dataset.py
DATAFILE_patient_list = 'C:/DATA/Master patient list for FDA 9-20-21_Seyed_v2.csv'
DATAFILE_dataset_features = 'C:/DATA/Aperio_datasetInfo_v2.csv'
OUTPUTDIR_WIN = 'C:/DATA/'
OUTPUTDIR_UBUNTU = '/home/seyedm.mousavikahaki/Documents/'
DATASETNAME = 'Aperio'
# TissueSegmentation2.py
SegmentationINPUTDIR = '/SubData/'
SegmentationOUTPUTDIR = 'C:/DATA/extracted_segmented/'
SegmentationTEMPDIR = "C:/DATA/tmp/"
# AnnotationExtraction
AnnotationExtractionSave_dir= '/home/seyedm.mousavikahaki/Documents/wxc4/extracted'
AnnotationExtractionSave_WSIs_ = '/home/seyedm.mousavikahaki/Documents/wxc4/*.svs'
AnnotationExtractionSave_WSIs_win = 'C:/DATA/Washu-Aperio/*.svs'
# Patch Generator
PatchGenerator_png_extract = True
PatchGenerator_hdf5 = True
PatchGenerator_open_dataset=True
PatchGenerator_Aplha = 2
PatchGenerator_INPUTDIR = 'extracted_segmented/'
PatchGenerator_OUTPUTDIR = 'C:/DATA/extracted_cutted/'
PatchGenerator_Number_of_Patches = 300
PatchGenerator_input_x = 256    
PatchGenerator_input_y = 256