#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */paind-project/process_image.py
# 
#
#                                                                            
# PROGRAMMER: Mark Rode
# DATE CREATED: November 16 2018     
# PURPOSE: Preprocesses image for Neural Network

# Import Python modules
from PIL import Image
from torchvision import transforms

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # open image
    img_pil = Image.open(image)
    # define transforms
    prep_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    # return transformed image
    return prep_image(img_pil)