ValidPath
================================================
The Whole Slide Image Processing and Machine Learning Performance Assessment Tool is a software program written in Python for analyzing whole slide images (WSIs), assisting pathologists in the assessment of machine learning (ML) results, and assessment of ML performance. The tool currently contains three modules that accept WSIs to generate image patches for AI/ML models, accept image patches (e.g., ML detected ROIs) to generate an Aperio ImageScope annotation file for validation of ML model results by pathologists, and accept outputs of ML models to generate performance results and their confidence intervals.


The tool provides the following components:

•	WSI handler for whole slide image processing and analysis

•	Extraction of pathologist annotations and extraction of image patches from the annotated areas of the whole slide images

•	Mapping ROIs onto WSI viewable by the Aperio ImageScope for the pathologist validation

•	Performance assessment of the ML results and statistical analysis

.. toctree::
   :maxdepth: 5

   installation
   inputrequirements
   WSI
   assessment
   
