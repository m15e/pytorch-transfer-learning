#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */paind-project/predict.py
# 
#
#                                                                            
# PROGRAMMER: Mark Rode
# DATE CREATED: November 15 2018     
# PURPOSE: Trains a model to classify images using a pretrained CNN model, compares these
#          classifications to the true identity of the flowers in the images, and
#          summarizes how well the CNN performed on the image classification task. 
#
#   Example call:
#    python train.py data_directory
#   Options:
#   Choose architecture: python train.py data_dir --arch "vgg13"
#   Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#   Use GPU for training: python train.py data_dir --gpu
##

# Import python modules - there may be duplicates but python only loads modules once 
from time import time, sleep
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import json
import os
# TODO: Can we use matplotlib from commandline
# import matplotlib.pyplot as plt
import copy
from PIL import Image


# Import functions needed for this program
from get_input_args import get_input_args
from get_data import get_data
from map_data import map_data
from load_model import load_model
from train_model import train_model


# Main program function defined below
def main():
    
    
    # get arguments from command line
    in_arg = get_input_args()
    
    # print that GPU is being used to console
    if in_arg.gpu == True and torch.cuda.is_available():
        print('Using CUDA')
    
    # get dataloaders from data directory
    dataloaders, class_names, dataset_sizes, class_to_idx  = get_data(in_arg.data)
    
    
    # get file with category mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # load pretrained model
    model = load_model(in_arg.arch, class_names, in_arg.hidden_units)
    # print model for dev purposes
    print('Test log: \n', model)
    # set loss function, optimizer 
    critereon = nn.CrossEntropyLoss()
    
    # only train the last layers of model
    if in_arg.arch == 'vgg' or in_arg.arch == 'alexnet':  
        optimizer_ft = optim.SGD(model.classifier.parameters(), lr=in_arg.learning_rate, momentum=0.9)
    else:
        optimizer_ft = optim.SGD(model.fc.parameters(), lr=in_arg.learning_rate, momentum=0.9)
    # dynamic learning rate    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    
    # train model and print results
    model = train_model(model, dataloaders, dataset_sizes,critereon, optimizer_ft, exp_lr_scheduler, in_arg.epochs, use_gpu=in_arg.gpu)    
    
    # set class to index mapping for future model rebuild
    model.class_to_idx = class_to_idx
    
    # create checkpoint for rebuilding model
    checkpoint = { 'state_dict': model.state_dict(), 'class_to_idx': model.class_to_idx, 'epochs': in_arg.epochs, 'optimizer': optimizer_ft.state_dict()  }
    # save architecture and last layers
    if in_arg.arch == 'resnet':
        checkpoint.update(fc = model.fc)
        checkpoint.update(reload = 'resnet50')
        
    elif in_arg.arch == 'alexnet' or in_arg.arch == 'vgg':
        checkpoint.update(classifier = model.classifier)
        
    if in_arg.arch == 'alexnet':
        checkpoint.update(reload = 'alexnet')
    if in_arg.arch == 'vgg':
        checkpoint.update(reload = 'vgg16')

    
    # create save directory if user inputs --save_dir option
    if in_arg.save_dir != '.':
        if not os.path.exists(in_arg.save_dir):
            os.makedirs(in_arg.save_dir)
        torch.save(checkpoint, '{}/checkpoint.pth'.format(in_arg.save_dir))
        # print model location
        print('\nModel saved to path: {}/checkpoint.pth'.format(in_arg.save_dir))
    # save results
    else:
        torch.save(checkpoint, 'checkpoint.pth')
        # print model location
        print('\nModel saved to path:rm - checkpoint.pth') 
        
    
    
    
# Call to main function to run program
if __name__ == '__main__':
    main()