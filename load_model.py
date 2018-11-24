#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */aipnd-project/load_model.py
#                                                                             
# PROGRAMMER: Mark Rode
# DATE CREATED: November 15 2018  
# PURPOSE: Loads pretrained model and sets output features

# import modules
import torch
from torchvision import models
import torch.nn as nn
from collections import OrderedDict

def load_model(arch, class_names, hidden_units = 0):
    
    # print string to make command line output easier to read
    print('\n' + '-' * 18)
    
    # Vgg
    if arch == 'vgg':

        print('Loading vgg16...')

        # load pretrained model
        model = models.vgg16(pretrained=True)
        # freeze model parameters
        for param in model.features.parameters():
            param.requires_grad = False
        # set in features
        in_ftrs = model.classifier[6].in_features
        # set hidden units if specified
        if hidden_units != 0:
            aug_layer = nn.Sequential(OrderedDict([
                                      ('hidden_layer', nn.Linear(in_ftrs, hidden_units)),
                                      ('relu', nn.ReLU()),
                                      ('output', nn.Linear(hidden_units, len(class_names)))
                                      ]))
            model.classifier[6] = aug_layer  
        else:
            model.classifier[6] = nn.Linear(in_ftrs, len(class_names))


    # Resnet
    elif arch == 'resnet':
        print('Loading resnet50...')
        # load pretrained model
        model = models.resnet50(pretrained=True)
        # freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        # set in features
        in_ftrs = model.fc.in_features
        # set hidden units if specified
        if hidden_units != 0:
            aug_layer = nn.Sequential(OrderedDict([
                              ('hidden_layer', nn.Linear(in_ftrs, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('output', nn.Linear(hidden_units, len(class_names)))
                              ]))
            model.fc = aug_layer 
        else:
            model.fc = nn.Linear(in_ftrs, len(class_names))
    # Alexnet
    elif arch == 'alexnet':

        print('Loading alexnet...')

        # load pretrained model
        model = models.alexnet(pretrained=True)
        # freeze model parameters
        for param in model.features.parameters():
            param.requires_grad = False
        # set in features
        in_ftrs = model.classifier[6].in_features
        # set hidden units if specified
        if hidden_units != 0:
            aug_layer = nn.Sequential(OrderedDict([
                              ('hidden_layer', nn.Linear(in_ftrs, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('output', nn.Linear(hidden_units, len(class_names)))
                              ]))
            model.classifier[6] = aug_layer 
        else:
            model.classifier[6] = nn.Linear(in_ftrs, len(class_names))
    # If people choose a model other than alexnet, resnet or vgg - print message
    else:
        print('Invalid model name, model name must be vgg, alexnet or resnet. exiting...')
    
    # Print model architecture
    print('\nCommencing training with model:\n' + '\n\n', model, '\n')
    return model