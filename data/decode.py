# Decode segmentation masks and save them as npy files in a mask folder
# Run this once only. It should take <5mins
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
tqdm.pandas()
from PIL import Image
import sys
import os
import cv2
import pdb

# from util import constants as C
IMAGE_SIZE = 224

# Reference: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    height, width = shape
    if mask_rle==mask_rle:
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(height*width, dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
    else: # If segmentation is NaN, just return an array of 0s
        img = np.zeros(height*width, dtype=np.uint8)
    return img.reshape(shape)  # Needed to align to RLE direction

if __name__ == '__main__':
    # usage: python decode.py [masks_folder_path] [combined_csv_path] [final_csv_path] [downsample frac]
    masks_folder_path = sys.argv[1]
    combined_csv_path = sys.argv[2]
    final_csv_path = sys.argv[3]
    downsample = float(sys.argv[4])

    # Make masks folder path if it doesn't already exist
    if not os.path.exists(masks_folder_path):
        os.mkdir(masks_folder_path)

    # Read in the combined df
    combined_df = pd.read_csv(combined_csv_path, keep_default_na=False, index_col=0)

    #Downsample some empty images
    if downsample != 1:
        combined_df['empty'] = combined_df.apply(lambda x: 1 if x.segmentation == '' else 0, axis = 1)
        combined_df['empty_image'] = combined_df.groupby(['id'])['empty'].transform(lambda x: sum(x))
        empty_df = combined_df[combined_df['empty_image']== 3]
        masks_df = combined_df[combined_df['empty_image']< 3]
        
        random_select = empty_df[empty_df['class'] == 'large_bowel'].sample(frac = downsample)['id']
        empty_df = pd.merge(random_select, empty_df, how = 'left', on = 'id')

        combined_df = pd.concat([masks_df, empty_df])

    classes = ['small_bowel', 'large_bowel', 'stomach'] # mask classes
    mask_paths = []
    
    # Decode!
    case_ids = combined_df['id'].unique()
    combined_df.set_index(['id', 'class'], inplace=True)
    for case_id in tqdm(case_ids):
        # make mask for each class and store in dict
        mask_dict = {}
        for mask_class in classes:
                        
            # identify row in df with relevant info for the case id and class
            id_class = combined_df.loc[case_id, mask_class]
            
            # decode the mask
            decoded_mask = rle_decode(id_class['segmentation'], (id_class['slice_height'], id_class['slice_width'])) 

            decoded_mask = cv2.resize(decoded_mask, (IMAGE_SIZE, IMAGE_SIZE))
            
            # store decoded mask in dictionary
            mask_dict[mask_class] = decoded_mask
        
        case_mask = np.stack([mask_dict[c] for c in classes], axis=-1)
        mask_path = os.path.join(masks_folder_path, case_id + '.npy')
        np.save(mask_path, case_mask)
        mask_paths.append(mask_path)
    
    # save csv of mask paths
    combined_df.reset_index(inplace=True)

    mask_path_df = combined_df[['case', 'day', 'slice_id', 'image_path', 'pic_info', 'slice_height', 'slice_width', 'pixel_height', 'pixel_width']]
    mask_path_df.drop_duplicates(inplace=True)
    mask_path_df['mask_path'] = mask_paths

    mask_path_df.to_csv(final_csv_path + '.csv')