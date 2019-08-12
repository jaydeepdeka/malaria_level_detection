# -*- coding: utf-8 -*-
"""
Description: This script runs runs through the JSON files, picks the image path
and co-ordinates, crops the cell images and saves it to the category files 
accordingly.

@author: jaydeep
"""
import os
import pandas as pd
import cv2
from pathlib import Path

dataset_path = Path(r'E:\Class_Notes_Sem2\ADM\Project\malaria-bounding-boxes\malaria')

def main():
    dataset_df = {}
    # Read the JSON contents to a dataframe
    for p,f,files in os.walk(dataset_path):
        for file in files:
            if file.endswith('json'):
                    dataset_df[file.strip('.json')] = pd.read_json(os.path.join(p, file))
    
    # Extract the path, objects and categories if the images
    object_df = {}
    for types in dataset_df.keys():
        dataset_df[types]['path'] = dataset_df[types]['image'].map(lambda x: dataset_path / x['pathname'][1:])
        dataset_df[types]['image_exists'] = dataset_df[types]['path'].map(lambda x: x.exists())
        object_df[types] = pd.DataFrame([dict(image=c_row['path'], **c_item) \
                 for _, c_row in dataset_df[types].iterrows() for c_item in c_row['objects']])
    
    # Create a folder for each category
    try:
        for types in object_df.keys():
            for category in (object_df[types]['category']).unique():
                os.mkdir(dataset_path / category)
    except:
        print("Folder already created")
    
    # Crop out the images using the co-ordinates
    for types in object_df.keys():
        count = 1
        for index, row in object_df[types].iterrows():
            path = row['image']
            im = cv2.imread(str(path))
            category = row['category']
            min_val = row['bounding_box']['minimum']
            max_val = row['bounding_box']['maximum']
            crop_img = im[min_val['r']:max_val['r'], min_val['c']:max_val['c']]
            cv2.imwrite(r"E:\Class_Notes_Sem2\ADM\Project\malaria-bounding-boxes\malaria\{}\{}.jpg".format(category, count), crop_img)
            cv2.waitKey(2)
            count+=1
    print("Cropping done, {} images cropped".format(count))
    return

if __name__== '__main__':
    main()