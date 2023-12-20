Installation
****************************************
This installation guide provides a detailed explanation and step by step guide to install packages required for SlidePro toolbox.

Required Packages
--------------------------------------------------------
In order to use ValidPath, you need to install some python packages. It is recommended to install the same version specified in this section (and in the requirement.txt). 

ValidPath was tested on the following environments : 

•	Linux System (Tested on Ubuntu 18.04.3 LTS) and Python 3.8.8

•	Windows 10 and Python 3.11.5

•	To install a python package with specific version of a package using pip, you can use the syntax “pip install package==version” in the command line. 

For example, in ValidPath we are using lxml which is one of the fastest and feature-rich libraries for processing XML and HTML in Python. 

To install lxml version 4.9.1, run the following command:

.. code-block:: console

	pip install lxml==4.9.1

Please follow the same procedure to install these python packages:

.. code-block:: console

	conda create --name WSIProcessing python= Python 3.8.8
	conda activate WSIProcessing
	pip install lxml==4.9.1
	pip install opencv-python==4.8.1.78
	pip install openslide-python==1.1.2
	pip install scikit-image==0.18.1
	pip install Shapely==1.7.1
	pip install sharepy==2.0.0
	pip install matplotlib==3.6.2 
	pip install Pillow==9.3.0
	pip install tifffile==2022.10.10
	pip install mpmath==1.2.1
	pip install h5py
	pip install    scikit-learn
	pip  install openpyxl
	pip install pandas    
	
	
Alternatively, the required packages can be installed at once, rather than installing them one by one, using the following command:

.. code-block:: console

	pip install –r requirements.txt

For the full list of the requirements, please see the requirement.txt file in the project root directory 

In order to check the current package version installed on you system, you can use “pip freeze” or “.___version___” as follows:

.. code-block:: console

	pip freeze | findstr lxml
	
or 

.. code-block:: console

	import lxml
	
	print(lxml.__version__)



Installation Using Anaconda
--------------------------------------------------------
Anaconda is a distribution of the Python and R programming languages for scientific computing, that aims to simplify package management and deployment. The distribution includes data-science packages suitable for Windows, Linux, and macOS. Wikipedia 

There are few steps to complete the installation. Firstly, you need to install Anaconda Navigator. This allows you to access to different Python IDEs and Python packages. When you install Anaconda Navigator, you may install your favorite IDEs such as Spider, PyCharm, and etc. You also will be able to create environment to have specific IDEs and Python packages for each project separately. Let’s start with Anaconda Navigator.    

Anaconda Navigator 

In order to install Anaconda Navigator, download the Anaconda distribution from the following URL: 

https://www.anaconda.com/products/distribution

Installing ValidPath using Anaconda
--------------------------------------------------------

Open a terminal window.

.. code-block:: console

    $ cd ValidPath ROOT DIRECTORY
	
Download a complete copy of the ** ValidPath **.

.. code-block:: console

    $ git clone https://github.com/DIDSR/wsi_processing_toolbox
	
Change directory to ValidPath

.. code-block:: console

    $ cd ValidPath
	
Create virtual environment for ** ValidPath** using

.. code-block:: console

    $ conda env create -f requirements.dev.conda.yml
	
.. code-block:: console
	
	$ conda activate ValidPath-dev
	
or

.. code-block:: console

    $ conda create -n ValidPath python=3.8 

.. code-block:: console

    $ conda activate ValidPath

.. code-block:: console

    $ pip install -r requirements.txt
	

To use the packages installed in the environment, run the command:

.. code-block:: console

    $ conda activate ValidPath-dev
	

Direct Installation of ValidPath
--------------------------------------------------------

You can install required packages and then use pip to install the ValidPath.

Windows

1. Download OpenSlide binaries from `this page <https://openslide.org/download/>`_. Extract the folder and add ``bin`` and ``lib`` subdirectories to
Windows `system path <https://docs.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v=office.14)>`_.

2. Install OpenSlide. The easiest way is to install OpenSlide is through pip
using

.. code-block:: console

    C:\> pip install OpenSlide

3. Install
ValidPath.

.. code-block:: console

    C:\> pip install ValidPath

Linux (Ubuntu)

On Linux the prerequisite software can be installed using the command

.. code-block:: console

    $ apt-get -y install libopenjp2-7-dev libopenjp2-tools openslide-tools


From Source
--------------------------------------------------------

The source code of the slidepro toolbox can be accessed from the GitHub.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/mousavikahaki/ValidPath.git
	
after downloading the source code of the slidepro toolbox, you can install it using the following command:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/mousavikahaki/ValidPath.git