.. ValidPath documentation master file, created by
   sphinx-quickstart on Wed Nov  9 10:26:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
============================================================================================================

About ValidPath
--------------------------------------------------------
The Whole Slide Image Processing and Machine Learning Performance Assessment Tool is a software program written in Python for analyzing whole slide images (WSIs), assisting pathologists in the assessment of machine learning (ML) results, and assessment of ML performance. The tool currently contains three modules that accept WSIs to generate image patches for AI/ML models, accept image patches (e.g., ML detected ROIs) to generate an Aperio ImageScope annotation file for validation of ML model results by pathologists, and accept outputs of ML models to generate performance results and their confidence intervals.


Modules
--------------------------------------------------------
There are several modules in this package including:

	1.	WSI handler: includes functions and classes for general WSI analysis such as read whole slide images, tissue segmentation, and normalization.
	2.	Annotation Extraction: this module includes several functions for processing annotations such as annotation extraction.
	3.	Patch Extraction: which assist pathologist and developers in extracting image patches from whole slide images region of interest.
	4.	Aperio ImageScope Annotation File Generator: to enable pathologist validation of the AI/ML results.
	5.	Performance Assessment: to assess the performance of ML models in classification tasks.

To see a demo of the functions in this toolbox, please refer to the Jupyter Notebooks files in the root folder of this package.

	•	01_read_wsi.ipynb_

	•	02_annotation_extraction.ipynb_

	•	03_patch_extraction.ipynb_

	•	4_annotation_generator.ipynb_

	•	05_performance_assessment.ipynb_
    
.. _01_read_wsi.ipynb: https://github.com/mousavikahaki/ValidPath/blob/main/01_read_wsi.ipynb

.. _02_annotation_extraction.ipynb: https://github.com/mousavikahaki/ValidPath/blob/main/02_annotation_extraction.ipynb

.. _03_patch_extraction.ipynb: https://github.com/mousavikahaki/ValidPath/blob/main/03_patch_extraction.ipynb

.. _4_annotation_generator.ipynb: https://github.com/mousavikahaki/ValidPath/blob/main/4_annotation_generator.ipynb

.. _05_performance_assessment.ipynb: https://github.com/mousavikahaki/ValidPath/blob/main/05_performance_assessment.ipynb

.. toctree::
   :hidden:

   self

.. toctree::
	:maxdepth: 3
	:titlesonly:
   
   
   installation
   inputrequirements
   WSI
   annotation
   patch
   ann_generator
   assessment



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
