#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */aipnd-project/get_data.py
#                                                                             
# PROGRAMMER: Mark Rode
# DATE CREATED: November 15 2018  
# PURPOSE: Create a function that defines data and transforms for training 
#          from the user using the torchvision transforms module.

# import modules
import torch
import torchvision
from torchvision import datasets, models, transforms
import os


def get_data(data_dir):
    # set directories
    data_dir = data_dir
    TRAIN = 'train'
    VAL = 'valid'
    # define datatransformations for training and validation sets
    data_transforms = {
        TRAIN: transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ]), 
        VAL: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    }
    # load datasets using torchvision datasets ImageFolder module
    image_datasets = { 
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            transform=data_transforms[x]
        )
        for x in [TRAIN, VAL]
    }  
    # define dataloaders for training
    dataloaders = { 
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
        for x in [TRAIN, VAL]
    }
    # provide dataset description
    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL]}
    # print data description
    print('\nInput data:\n' + '-' * 18 + '\n')
    for x in [TRAIN, VAL]:
        print('{} images under {}'.format(dataset_sizes[x], x))
    class_names = image_datasets[TRAIN].classes
    class_to_idx = image_datasets[TRAIN].class_to_idx
    
    # return dataloaders and class names
    return dataloaders, class_names, dataset_sizes, class_to_idx