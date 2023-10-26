from setuptools import setup, find_packages
import codecs
import os
import sys
from pathlib import Path

with open("README.md") as readme_file:
    readme = readme_file.read()

install_requires = [
    line
    for line in Path("requirements.txt").read_text().splitlines()
    if line and line[0] not in ("-", "#")
]

VERSION = '0.0.1'
DESCRIPTION = 'Processing Digital Pathology Data'
LONG_DESCRIPTION = 'A package that Allows processing of digital pathology whole slide image data'

# Setting up
setup(
    name="slidepro",
    version=VERSION,
    author="Seyed Mostafa Mousavi Kahaki",
    author_email="<mousavikahaki@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['opencv-python', 'pyautogui', 'pyaudio'],
    keywords=['python', 'whole slide', 'image', 'pathology', 'WSI', 'medical image'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)