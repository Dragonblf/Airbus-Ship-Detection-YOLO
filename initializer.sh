#!/bin/bash

# Initializes and download all neccessary files
# for the competition 'Airbus Ship Detection'


# 1. Convert VOC-Annoations to the needed YoloV3 format
python framework/convert/convert_annotations.py --annotations=$1 --labels=$2 --output=$3

# 2. Download official yolov3.weigths
wget https://pjreddie.com/media/files/yolov3.weights -P framework/darknet/weights/yolov3.weights

# 3. Convert config and weights to Keras h5 model
python framework/convert/convert.py framework/darknet/configs/yolov3.cfg framework/darknet/weights/yolov3.weights framework/model_data/yolo3.h5
rm framework/darknet/weights/yolov3.weights