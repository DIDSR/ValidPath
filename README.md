<p align="center">
  <h1 align="center">This tool is under developement</h1>
</p>

<p align="center">
  <img src="img/SlideProLogo.png">
</p>

<h1 align="center"> </h1>
<p align="center">
  <h1 align="center">Digital Pathology Toolbox developed at DIDSR</h1>
</p>
<p align="center">
  <img src="img/CAH_Carcinoma.png">
</p>





## Getting Started

### General Information
**`WSIToolbox`** is a python-based package for developers and scientist who are interested in digital pathology. The main goal of developing this tool is to help stakeholders, graduate students, and pathologist to speed up their projects.  For more information please contact: **[seyed.kahaki@fda.hhs.gov](mailto:seyed.kahaki@fda.hhs.gov)**.

We are continuously working on this toolbox, and we welcome any contributions.

### Modules
There are several modules in this package including
1.	WSI Handler: includes functions and classes for general wsi analysis such as read whole slide images, tissue segmentation, and normalization.
2.	Annotation Extraction: this module includes several functions for processing annotations such as annotation extraction.
3.	Patch Extraction: which assist pathologist and developers in extracting image patches from whole slide images region of interest.
4.	Annotation File Generator: Mapping back the ROIs into the image scope visualizer for the pathologist validation process
5.	Performance Assessment: including different modules for assessing the performance of ML models.

### Information for Developers
Code Documentation:
https://htmlpreview.github.io/?https://github.com/DIDSR/wsi_processing_toolbox/blob/main/docs/_build/html/index.html
Please refer to the code documentation and email  **[seyed.kahaki@fda.hhs.gov](mailto:seyed.kahaki@fda.hhs.gov)** if you would like to collaborate on this project.


### Testing Examples
1. WSI Reader: Read Whole slide Image, Extract WSI Regions
2. Extract Annotations: Extract annotations and masks from Whole Slide Images
3. Patch Extraction: Extract patch images from annotated regions
4. Color Normalization: Color normalization of extracted patches
5. Annotation Generator: (To be Added)
6. Performance Assessmet: including different methods for assessing the performance of ML models

## Installation
This section will help you to install the packages WSIToolbox.

### Install Python package

If you wish to use our python package, perhaps without developing them further, run the command pip install WSIToolbox or pip install --ignore-installed --upgrade WSIToolbox to upgrade from an existing installation (This will be enabled when the first version is ready and submitted to pypi).

Detailed installation instructions can be found in the [documentation](Link to Installation Guide dependencies).

To understand better how the programs work, study the jupyter notebooks referred to under the heading **Examples Taster**.

### Pre-requirements

In order to use WSIToolbox, you need to install some python package. It is recommended to install the same version specified in this section (and in the requirement.txt). WSIToolbox was tested on the following environment: 
- Linux System (Tested on Ubuntu 18.04.3 LTS)
- Python 3.8
  
To install a python package with specific version of a package using pip, you can use the syntax “pip install package==version” in the command line. For example in WSIToolbox we are using lxml which is one of the fastest and feature-rich libraries for processing XML and HTML in Python. To install lxml version 4.9.1, run the following command:
```sh
pip install lxml==4.9.1
```
Please follow the same procedure to install these python packages:
•	lxml==4.9.1
•	opencv-python==4.8.1.78
•	openslide-python==1.1.2
•	scikit-image==0.18.1
•	Shapely==1.7.1
•	sharepy==2.0.0
•	matplotlib==3.6.2 
•	Pillow==9.3.0
•	tifffile==2022.10.10
•	mpmath==1.2.1
•	random
•	glob
•	pandas
•	numpy
•	For the full list of the requirements, please see the requirement.txt file in the project root directory 

  
In order to check the current package version installed on you system, you can use “pip freeze” or “.___version___” as follows:
```sh
  pip freeze | findstr lxml
```
or 
```sh
  import lxml
  print(lxml.__version__)
```



### Prepare for development (this is optional)

Prepare a computer as a convenient platform for further development of the Python package WSIToolbox and related programs as follows.
1.	Install the dependencies based on this guide (Link to Installation Guide dependencies)
2.	Open a terminal window
```sh
    $ cd WSIToolbox ROOT DIRECTORY
```
4.	Download a complete copy of the ** WSIToolbox **.
```sh
  $ git clone https://github.com/DIDSR/wsi_processing_toolbox
```
5.	Change directory to WSIToolbox
```sh
  $ cd WSIToolbox
```
6.	Create virtual environment for ** WSIToolbox** using
```sh
  $ conda env create -f requirements.dev.conda.yml
  $ conda activate WSIToolbox-dev
```
or
```sh
  $ conda create -n WSIToolbox-dev python=3.8 
  $ conda activate WSIToolbox-dev
  $ pip install -r requirements_dev.txt
```
7.	To use the packages installed in the environment, run the command:
```sh
$ conda activate WSIToolbox-dev
```


### Cite this repository

If you find that WSIToolbox is useful or if you use it in your project, please consider citing this paper:

```
@article{
    Pocock2022,
    author = {Seyed M. M. Kahaki, U.S. Food and Drug Administration (United States); Ian S. Hagemann, Washington Univ. School of Medicine in St. Louis (United States); Kenny Cha, Christopher J. Trindade, Nicholas Petrick, Weijie Chen, U.S. Food and Drug Administration (United States)},
    doi = {TBD},
    issn = {TBD},
    journal = {SPIE Medical Imaging},
    month = {feb},
    number = {1},
    pages = {1},
    publisher = {SPIE Medical Imaging},
    title = {{Weakly supervised deep learning for predicting the response to hormonal treatment of women with atypical endometrial hyperplasia: a feasibility study}},
    url = {https://spie.org/medical-imaging/presentation/Weakly-supervised-deep-learning-for-predicting-the-response-to-hormonal/12471-31},
    volume = {2},
    year = {2023}
}
```

### Auxiliary Files

Pre-trained models and their weights can be accessed from https://github.com/DIDSR/wsi_processing_toolbox/tree/main/models.


### Acknowledgment 
This project was supported in part by an appointment to the ORISE Research Participation Program at the Center for Devices and Radiological Health, U.S. Food and Drug Administration, administered by the Oak Ridge Institute for Science and Education through an interagency agreement between the U.S. Department of Energy and FDA/CDRH.

### License
The SlideProToolbox code is released under the XXX licence. The full text for the licence can be accessed from [LICENSE](LINK to Licence).
