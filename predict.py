#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */paind-project/predict.py
# 
#
#                                                                            
# PROGRAMMER: Mark Rode
# DATE CREATED: November 16 2018     
# PURPOSE: Loads a pretrained CNN model and an image as inputs and returns the flower name and class probability
#
#   Example call:
#   python predict.py /path/to/image checkpoint
#   Options:
#   Return top K most likely classes: python predict.py /path/to/image checkpoint --top_k 3
#   Change mapping of categories to real names: python predict.py /path/to/image checkpoint --category_names cat_to_name.json 
# 
#   Use GPU for training: python predict.py /path/to/image checkpoint --gpu
##

# Imports
import torch

# Import functions
from get_predict_inputs import get_predict_inputs
from process_image import process_image
import torchvision
from torchvision import models
import torch.nn.functional as F
import json

# Main program function defined below
def main():
    
    # get inputs
    in_arg = get_predict_inputs()
    
    # parse commandline inputs 
    
    # turn parsed arguments into variables
    image = in_arg.image
    checkpoint = in_arg.check_point
    top_k = in_arg.top_k
    cat_to_name_file = in_arg.category_names
    
    
    # Provision in case GPU trained model is to be predicted on CPU
    if in_arg.gpu == False:
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint)
    # Get model and state_dict from checkpoint
    model = getattr(torchvision.models, checkpoint['reload'])(pretrained=True)
    
    # set last layers depending on architecture
    if checkpoint['reload'] == 'vgg16' or checkpoint['reload'] == 'alexnet':  
        # freeze layers
        for param in model.features.parameters():
            param.requires_grad = False
        # replace final layers
        model.classifier = checkpoint['classifier']
    if checkpoint['reload'] == 'resnet50':
        # freeze layers
        for param in model.parameters():
            param.requires_grad = False
        # replace final layers
        model.fc = checkpoint['fc']
    
    # load model training data
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    # make prediction
    ###########################
    
    # set to GPU if specified
    if in_arg.gpu and torch.cuda.is_available():
        print('Using CUDA')
        model.to('cuda')
        # process image
        model_image = process_image(image).unsqueeze_(0).cuda()
    else:
        # process image
        model_image = process_image(image).unsqueeze_(0)
    
    # make prediction
    output = model(model_image)
    # convert output to probabilities
    output = F.softmax(output, dim=1)
    
    # list top_k predictions
    outputs = list(torch.topk(output, top_k))
    
    # convert probs tuple to numpy array, convert classes to list of strings
    probs = outputs[0].cpu().detach().numpy()[0]
    class_indices = outputs[1].cpu().detach().numpy()[0]
    
    # convert indices to classes
    index_to_class = { val: key for key, val in model.class_to_idx.items() }
    top_classes = [index_to_class[each] for each in class_indices]
    
    # turn numpy probs into list for printing
    probs_list = [str(x) for x in list(probs)]
    # turn classes into list for real name conversion
    class_list = [str(x) for x in list(top_classes)]
    
    # open mapping file
    with open(cat_to_name_file, 'r') as f:
        cat_to_name = json.load(f)
    # turn classes into names
    class_map = []
    for c in class_list:
        # provision because indexing of mapping file and prediction classes is not equal
        if c == '0':
            class_map.append('pink primrose')
        # loop to map classes to names
        for key, value in cat_to_name.items():
            if c == key:
                class_map.append(value)
                
            
    # print predictions    
    print('\nModel prediction(s):\n' + '-' * 18 + '\n' )
    for n, c, p in zip(class_map, top_classes, probs_list):
        
        print('{}, class[{}]: {:.4f}%'.format(n.capitalize() , c, float(p)*100))
    print('\n')
# Call to main function to run program

if __name__ == '__main__':
    main()