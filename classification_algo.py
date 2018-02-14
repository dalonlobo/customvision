#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:37:01 2018

@author: dalonlobo
"""
from __future__ import absolute_import, division, \
                        print_function, unicode_literals
import os
import sys
import argparse
import logging
import glob
import shutil
import json
# Install azure package using:
# pip install "git+https://github.com/Azure/azure-sdk-for-python#egg=azure-cognitiveservices-vision-customvision&subdirectory=azure-cognitiveservices-vision-customvision"
# If you encounter a Filename too long error, make sure you have long path support in git enabled.
# git config --system core.longpaths true
from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
from azure.cognitiveservices.vision.customvision.prediction.prediction_endpoint import models

logger = logging.getLogger("__main__") 

def move_to_folder(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    shutil.move(src, dst)

if __name__ == "__main__":
    """
    Classifies the image into its respective class,
    Make sure to insert the keys in the config file.
    Only jpg, png and bmp image extensions can be used
    Size of each image should not exceed 4mb
    :input:
        conf_file: Configuration json file, check the sample_conf.json
        src_path: Path the image files 
        dest_path: Path to save the files
    :output:
        Saves the images in their respective categories folder
    """
    logs_path = os.path.basename(__file__) + ".logs"
    logging.basicConfig(filename=logs_path,
        filemode='a',
        format='%(asctime)s [%(name)s:%(levelname)s] [%(filename)s:%(funcName)s] #%(lineno)d: %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    try:
        print("Logs are in ", os.path.abspath(logs_path), file=sys.stderr)
        print("Run the following command to view logs:\n", file=sys.stderr)
        print("tail -f {}".format(os.path.abspath(logs_path)), file=sys.stderr)
        parser = argparse.ArgumentParser(description="Classifier")
        parser.add_argument('--conf_file', type=str,  
                            help='Path to the json configuration file')
        parser.add_argument('--src_folder', type=str,  
                            help='Path to the folder which contains images to be classified')
        parser.add_argument('--dest_folder', type=str,  
                            help='Path to the folder where the output images are stored')
        args = parser.parse_args()
    
        logger.info("#########################")
        logger.info(".....Exiting program.....")
        logger.info("#########################")
        if not os.path.exists(args.dest_folder):
            logger.debug("Creating the destination folder")
            os.mkdir(args.dest_folder)
        
        # Now there is a trained endpoint, it can be used to make a prediction
        with open(args.conf_file, "rb") as f:
            keys = json.load(f)
        project_id = keys["project_id"]
        prediction_key = keys["prediction_key"]
        logger.info("Initializing the predictor")
        predictor = prediction_endpoint.PredictionEndpoint(prediction_key)
        image_extensions = ["jpg", "png", "bmp"]
        for ext in image_extensions:
            for file_name in glob.glob(args.src_folder + os.path.sep + "*." + ext):
                logger.info("Calling the prediction on:")
                logger.info(file_name)
                # Open the image and get back the prediction results.
                with open(file_name, mode="rb") as test_data:
                    results = predictor.predict_image(project_id, test_data.read())
                # Display the results.
                for prediction in results.predictions:
                    logger.debug("\t" + prediction.tag + \
                           ": {0:.2f}%".format(prediction.probability * 100))
                highest_tag = results.predictions[0].tag
                move_to_folder(file_name, args.dest_folder + os.path.sep + highest_tag)
    except Exception as e:
        logger.exception(e)
        print("Exception has occured, check the logs", file=sys.stderr)
    finally:
        logger.info("#########################")
        logger.info(".....Exiting program.....")
        logger.info("#########################")
        print("#########################", file=sys.stderr)
        print(".....Exiting program.....", file=sys.stderr)
        print("#########################", file=sys.stderr)