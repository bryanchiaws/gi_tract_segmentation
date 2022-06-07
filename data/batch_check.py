# Check batches
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
tqdm.pandas()
from PIL import Image
import random
import sys
import os
import cv2
import pdb
import shutil

def batch_check(dataset, num_batches):
    data = pd.read_csv(dataset)
    batch_list = random.sample(range(data['batch'].min(), data['batch'].max()+1), num_batches)

    # Make masks folder path if it doesn't already exist
    if not os.path.exists("batch_check"):
        os.mkdir("batch_check")


    for b in batch_list:
        subset = data[data['batch']==b]
        subset['new_name'] = data['case'] + "_" + data['day'] + "_slice" + data['slice_id'].astype(str)
        subset = subset[['image_path', 'new_name', 'mask_path']]
        path = os.path.join("batch_check", "batch" + str(b))

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)           # Removes all the subdirectories!
            os.makedirs(path)
        
        empty_count = 0 
        for _, row in subset.iterrows():
            image, name, mask_path = row
            # save images in new folder
            shutil.copy(image, os.path.join(path, name + ".png"))

            # check if mask is all black
            mask = np.load(mask_path)
            if mask.sum() == 0:
                empty_count += 1
        
        empty_pct = 100*empty_count / len(subset)
        print("For batch {}, {}% of masks are empty".format(b, empty_pct))


if __name__ == '__main__':
    # usage: python batch_check.py dataset_name.csv num_batches
    dataset = sys.argv[1]
    num_batches = int(sys.argv[2])

    batch_check(dataset, num_batches)


