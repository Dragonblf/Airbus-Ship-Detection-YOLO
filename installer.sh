#!/bin/bash

# Installs all needed libraries for this library.
# (Only with pip ... if you wish to use conda or something
#  else you have to do it yourself.)
#  --> Works only on Ubuntu

pip install graphviz>=0.8.4
pip install Keras>=2.2.0
pip install lxml>=4.1.1
pip install Markdown>=2.6.9
pip install matplotlib>=2.2.3
pip install numpy>=1.14.5
pip install opencv-python>=3.4.0.12
pip install pandas>=0.21.1
pip install Pillow>=4.3.0
pip install tensorboard>=1.6.0
pip install tensorflow>=1.3.0
pip install tensorflow-gpu>=1.4.0
pip install tqdm>=4.11.2
pip install albumentations>=0.1.2

apt-get install graphviz