import numpy as np
import pandas as pd
import random
from glob import glob
import os, shutil
from tqdm import tqdm
tqdm.pandas()
import time
from PIL import Image
import sys

def prepare_label_df(label_csv_path):
    df = pd.read_csv(label_csv_path)
    df["absent"] = df["segmentation"].map(lambda x: int(pd.isna(x))) # 1 means the organ is absent, 0 means its present
    df["case"] = df["id"].str.split('_').str[0]
    df["day"] = df["id"].str.split('_').str[1]
    df["slice_id"] = df["id"].str.split('_').str[3]
    return df

def prepare_image_df(dataset_folder_path, path_split):
    images = glob(dataset_folder_path + '/*/*/*/*.png')
    image_df = pd.DataFrame(images, columns=["image_path"])

    image_df["case"] = image_df["image_path"].str.split(path_split).str[1]
    image_df["day"] = image_df["image_path"].str.split(path_split).str[2].str.split('_').str[1]
    image_df["slice_id"] = image_df["image_path"].str.split(path_split).str[4].str.split('_').str[1]

    image_df["pic_info"] = image_df["image_path"].str.split(path_split).str[4]
    image_df["slice_height"] = image_df["pic_info"].str.split("_").str[2].astype(int)
    image_df["slice_width"] = image_df["pic_info"].str.split("_").str[3].astype(int)
    image_df["pixel_height"] = image_df["pic_info"].str.split("_").str[4].astype(float)
    image_df["pixel_width"] = image_df["pic_info"].str.split("_").str[5].str.split('.png').str[0].astype(float)
    return image_df

if __name__ == '__main__':
    # usage: python csv_process.py [label_csv_path] [dataset_folder_path] [output_csv_name] [path_splitter]
    label_csv_path = sys.argv[1]
    dataset_folder_path = sys.argv[2]
    output_csv_path = sys.argv[3]
    path_split = sys.argv[4]
    label_df = prepare_label_df(label_csv_path)
    image_df = prepare_image_df(dataset_folder_path, path_split)
    combined_df = pd.merge(label_df, image_df, how='left', on=['case','day','slice_id'])
    combined_df.to_csv(output_csv_path + ".csv")
    print("Combined CSV generated!")



    

