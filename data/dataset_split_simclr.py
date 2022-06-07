import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pdb
import numpy as np
import math

def pairing(odds, idx = ''):
    odds['dummy'] = 1
    odds['pair'] = odds.sort_values(['case','slice_id', 'day'],ascending=False).groupby(['case', 'day'])['slice_id'].shift()
    odds = odds[odds['slice_id'] % 2 == 1]
    odds['pair_idx'] = odds['dummy'].cumsum()
    odds['pair_idx'] = odds['pair_idx'].apply(lambda x: idx + str(x))

    #odds['pair'] = odds['pair'].astype('int')
    odds = pd.concat([odds[['case', 'slice_id', 'day', 'pair_idx', 'dummy']], odds[['case', 'pair', 'day', 'pair_idx', 'dummy']].rename({'pair': 'slice_id'}, axis = 'columns')])

    #Drop those that only have 1
    odds['pair_total'] = odds.groupby(['pair_idx'])['dummy'].sum()
    odds = odds[odds['pair_total'] != 2]
    return odds.drop(['pair_total', 'dummy'], axis = 'columns')

def dataset_split(dataset_path, output_folder, train_prop=0.7, val_prop=0.2, test_prop=0.1):
    # Default split is 70/20/10
    # Check if the splits add up to 1
    total = train_prop + val_prop + test_prop
    if abs(total - 1) > 0.0000001:
        print("Train, validation, and test proportions must add up to 1. Instead, they are", round(total, 3))

    # Create output folder if its not already created
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    data = pd.read_csv(dataset_path, index_col=0)

    #Find pairs
    pair_ids = pairing(data)

    data = pd.merge(pair_ids, data, on = ['case', 'slice_id', 'day'])
    data['dummy'] = data.groupby(['pair_idx'])['dummy'].cumsum()
    data1 = data[data['dummy'] == 1]

    #Random sort, make sure no more than one repeat in a batch
    min_check = 0
    while min_check <= 1:
        data1 = data1.sample(frac=1).reset_index(drop=True)
        data1['batch'] = data1.index // 16
        check = data1.groupby('batch')['case'].count()
        min_check = min(check)
    
    data2 = pd.merge(data[data['dummy'] == 2], data1[['pair_idx', 'batch']], on = 'pair_idx')
    data = pd.concat([data1, data2]).reset_index(drop=True).sort_values(['batch', 'pair_idx'])

    n_batches = max(data1['batch'])
    val_batch = int(math.floor(val_prop * n_batches))
    test_batch = int(math.floor(test_prop * n_batches))
    train_batch = n_batches - val_batch - test_batch
    
    # Split into train and val and test datasets
    train = data[data['batch'] <= train_batch]
    val = data[(data['batch'] > train_batch) & (data['batch'] <= train_batch + val_batch)]
    test = data[data['batch'] > train_batch + val_batch]

    # Output train, val, test datasets
    train.to_csv(os.path.join(output_folder, "train_dataset.csv"), index=False)
    val.to_csv(os.path.join(output_folder, "val_dataset.csv"), index=False)
    test.to_csv(os.path.join(output_folder, "test_dataset.csv"), index=False)

if __name__ == '__main__':
    # usage: python data/dataset_split.py [final.csv] [output folder] [train percent] [val percent] [test percent]
    dataset_path = sys.argv[1]
    output_folder = sys.argv[2]
    try:
        train_prop = int(sys.argv[3]) / 100
        val_prop = int(sys.argv[4]) / 100
        test_prop = int(sys.argv[5]) / 100
        #print(f"Using train-val-test split of {sys.argv[3]}%-{sys.argv[4]}%-{sys.argv[5]}%")
        dataset_split(dataset_path, output_folder, train_prop, val_prop, test_prop)
    except:
        #print("Using default train-val-test split of 70%-20%-10%")
        dataset_split(dataset_path, output_folder)