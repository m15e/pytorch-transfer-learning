#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */aipnd-project/load_model.py
#                                                                             
# PROGRAMMER: Mark Rode
# DATE CREATED: November 15 2018  
# PURPOSE: Trains model 

# import modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import copy
import numpy as np
import torchvision
import time

def train_model(model, dataloaders, dataset_sizes, critereon, optimizer, scheduler, num_epochs=10, use_gpu=False):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' *10)

        # Each epoch has a training and a validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train() # set model to training mode
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            steps = 0
            for inputs, labels in dataloaders[phase]:
                steps += 1
                if use_gpu:
                    model.to('cuda')
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    
                #zero the parameter gradients
                optimizer.zero_grad()
                
                # forward pass
                # track history if in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = critereon(outputs, labels)
                    
                    # backward pass + optimize if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # update stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                if steps % 10 == 0:    
                    print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                    
                    
                
                # save best version of model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            print()
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model