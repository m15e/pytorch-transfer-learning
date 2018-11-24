#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */aipnd-project/get_input_args.py
#                                                                             
# PROGRAMMER: Mark Rode
# DATE CREATED: November 15 2018  
# PURPOSE: Create a function that retrieves command line inputs 
#          from the user using the Argparse Python module.

# Import python modules
import argparse

def get_input_args():
    
    # Create parser
    parser = argparse.ArgumentParser()
    
    # Argument 1: folder containing images - required
    parser.add_argument('data', type = str, help = 'path to image folder')
    # Argument 2: set directory for checkpoints - optional
    parser.add_argument('--save_dir', type = str, default = '.', help = 'where to save trained model - defaults to current folder')
    # Argument 3: set model architecture - optional
    parser.add_argument('--arch', type = str, default = 'vgg', help = 'select model architecture - defaults to vgg(16)')
    # Argument 4: set hyperparameter learning rate - optional
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'set model learning rate - defaults to 0.001')
    # Argument 5: set hyperparameter hidden units - optional
    parser.add_argument('--hidden_units', type = int, default = 0, help = 'option to add a hidden layer with n units - defaults to 0')
    # Argument 6: set hyperparameter epochs - optional
    parser.add_argument('--epochs', type = int, default = 10, help = 'set number of epochs - defaults to 10')
    # Argument 7: set computing power - optional
    parser.add_argument('--gpu', help='use gpu power', action='store_true')
    
    # Return input args
    return parser.parse_args()