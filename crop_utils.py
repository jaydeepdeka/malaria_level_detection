# -*- coding: utf-8 -*-
"""
Description: This script runs reads the files test.JSON and train.JSON picks the image path
and co-ordinates, crops the cell images and saves it to the category folders 
accordingly.

@author: jaydeep
"""
import os
import pandas as pd
import cv2
from pathlib import Path

# Dataset path
dataset_path = Path(r'E:\Class_Notes_Sem2\ADM\Project\malaria-bounding-boxes\malaria')

def main():
    # Dataframe to store to serialise JSON
    dataset_df = {}
    # Read the JSON contents to the dataframe
    for p,f,files in os.walk(dataset_path):
        for file in files:
            if file.endswith('json'):
                    dataset_df[file.strip('.json')] = pd.read_json(os.path.join(p, file))
    
    # Extract the path, objects and categories if the images
    object_df = {}
    for types in dataset_df.keys():
        # Retrieve the pathnames
        dataset_df[types]['path'] = dataset_df[types]['image'].map(lambda x: dataset_path / x['pathname'][1:])
        # Check for image existence
        dataset_df[types]['image_exists'] = dataset_df[types]['path'].map(lambda x: x.exists())
        # For each image store paths and objects
        object_df[types] = pd.DataFrame([dict(image=c_row['path'], **c_item) \
                 for _, c_row in dataset_df[types].iterrows() for c_item in c_row['objects']])
    
    # Create a folder for each category
    try:
        for types in object_df.keys():
            for category in (object_df[types]['category']).unique():
                os.mkdir(dataset_path / category)
    except:
        print("Folder already exists")
    
    # Crop out the images using the co-ordinates
    for types in object_df.keys():
        count = 1
        # Iter through each row of the dataframe
        for index, row in object_df[types].iterrows():
            # Get the path
            path = row['image']
            # Read the image
            im = cv2.imread(str(path))
            # Get the category to store the cropped image later to folder
            category = row['category']
            # Get the lower left coordinates
            min_val = row['bounding_box']['minimum']
            # Get the upper right coordinates
            max_val = row['bounding_box']['maximum']
            # Crop Image
            crop_img = im[min_val['r']:max_val['r'], min_val['c']:max_val['c']]
            # Write to the folder
            cv2.imwrite(r"E:\Class_Notes_Sem2\ADM\Project\malaria-bounding-boxes\malaria\{}\{}.jpg".format(category, count), crop_img)
            cv2.waitKey(2)
            count+=1
    print("Cropping done, {} images cropped".format(count))
    return

if __name__== '__main__':
    main()