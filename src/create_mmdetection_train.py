import argparse
import os
import pickle
from multiprocessing import Pool
import json
import pandas as pd
from tqdm import tqdm
from utils import group2mmdetection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str)
    parser.add_argument('--output', type=str)
    return parser.parse_args()

def create_images_coco(filename):
    return {
        'file_name': str(filename),
        'height': 1024,
        'width': 1024,
        'id': int(filename.split('.', 1)[0], 16)   # image_id in annoatations function
        }

def create_df_anns_coco(filename, bbox, id): # id is unique to each annoationations dict
    return {
        'segmentation': [],  # if you have mask labels which we don't
        'area': int(bbox[2])*int(bbox[3]), # area of bbox
        'iscrowd': 0,
        'image_id': int(filename.split('.', 1)[0], 16) , # from hex string to int id in create_images_coco
        'bbox': bbox,
        'category_id': 1,
        'id': id  # unique to this
        }

# what are the id fields in both?

def main():
    args = parse_args()
    df_ann = pd.read_csv(args.annotation)
    
    # apply create_images_coco problem is many files with same image id need to get unique
    df_ann['bbox'] = df_ann['bbox'].apply(lambda x: json.loads(x))
    unique_imgs = df_ann.image_id.unique()
    img_coco = [create_images_coco(x) for x in unique_imgs] #1
    df_ann["ann_coco"] = df_ann.apply(lambda x: create_df_anns_coco(x['image_id'], x['bbox'], x.name), axis=1)
    ann_coco = df_ann["ann_coco"].to_list() #2
    categories_coco = [{'id':1, 'name':"wHeat"}] # 3 (ref Family Guy Stewie: wHeat thins)
    info = [{"description": "Wheat Dataset", "version": "1.0", "year": 2020}]

    coco_dataset = { 
        'info':info,
        'images': img_coco,
        'annotations': ann_coco,
        "categories": categories_coco
    }
    with open(args.output, 'wb') as f:
        pickle.dump(coco_dataset, f)



if __name__ == '__main__':
    main()
