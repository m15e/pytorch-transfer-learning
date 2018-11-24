#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */aipnd-project/map_data.py
#                                                                             
# PROGRAMMER: Mark Rode
# DATE CREATED: November 15 2018  
# PURPOSE: Create a function that maps data from json file
#          from the user using the torchvision transforms module.

def map_data(classes, map_file):
    # create empty list for map
    class_map = []
    # iterate through class
    for c in classes:
        # fixes mapping because map file count starts with 1, classes start with 0
        if c == '0':
            class_map.append(mapfile.get('1'))
        for key, value in map_file.items():
            if c == key:
                class_map.append(value)
    return class_map