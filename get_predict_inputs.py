#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */aipnd-project/get_predict_inputs.py
#                                                                             
# PROGRAMMER: Mark Rode
# DATE CREATED: November 15 2018  
# PURPOSE: Create a function that retrieves command line inputs 
#          from the user using the Argparse Python module.

# Import python modules
import argparse

def get_predict_inputs():
    
    # Create parser
    parser = argparse.ArgumentParser()
    
    # Argument 1: file containing image to predict - required
    parser.add_argument('image', type = str, help = 'path to image e.g. flowers/test/1/img_02491.jpg')
    # Argument 2: path to saved model - required
    parser.add_argument('check_point', type = str, help = 'path to checkpoint file e.g. vgg16_checkpoint.pth')
    # Argument 3: set how many most likely classes model should output - optional
    parser.add_argument('--top_k', type = int, default = 1, help = 'select how many classes model should output e.g. top_k 5')
    # Argument 4: set mapping of categories to rel names - optional
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'converts category numbers to category names')
    # Argument 5: set computing power - optional
    parser.add_argument('--gpu', help='use gpu power', action='store_true')
    
    # Return input args
    return parser.parse_args()