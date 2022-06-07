import numpy as np
import pandas as pd
import torch
from PIL import Image
#import torchvision.transforms as T
import torch.nn as nn
# from detectron2.data import DatasetMapper

from util import constants as C
from .transforms import get_transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import albumentations.augmentations as AA

import pdb
import cv2

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transforms=None, split='train', 
                augmentation=None, image_size=224, pretrained=False):
        try:
            self._df = pd.read_csv(dataset_path).sort_values(['batch', 'pair_idx']).reset_index(drop = True)
        except:
            self._df = pd.read_csv(dataset_path)
        #self._df = self._df.sample(frac = 0.15).reset_index() # Careful of index_col here
        self._image_path = self._df['image_path']
        self._mask_path = self._df['mask_path']      
        self._pretrained = pretrained
        self.augmentation = augmentation
        self._transforms = get_transforms(
            split=split,
            augmentation=augmentation,
            image_size=image_size
        )

    def get_batch_list(self):
        indices = list(self._df.index)
        lol = [indices[i:i+32] for i in range(0, len(indices), 32)]
        return lol

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):

        image = cv2.imread(self._image_path[index], cv2.IMREAD_UNCHANGED)
        image = (image - image.min())/(image.max() - image.min())*255.0 
        image = cv2.resize(image, (C.IMAGE_SIZE, C.IMAGE_SIZE))
        image = np.tile(image[...,None], [1, 1, 3])
        image = image.astype(np.float32) /255.

        mask = np.load(self._mask_path[index])

        mask = torch.tensor(mask.transpose(2, 0, 1), dtype = torch.float32)
        image = torch.tensor(image.transpose(2, 0, 1), dtype = torch.float32)

        #if self.augmentation != 'none':
        #    mask = self._transforms(mask)
        #    image = self._transforms(image)

        return image, mask

class SegmentationDemoDataset(SegmentationDataset):
    def __init__(self):
        super().__init__(dataset_path=C.TEST_DATASET_PATH)

class ImageDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_path=None, annotations=None, augmentations=None):
        self._image_path = image_path
        self._annotations = annotations
        self._mapper = DatasetMapper(is_train=True,
                                     image_format="RGB",
                                     augmentations=augmentations
                                     )

    def __len__(self):
        return len(self._annotations)

    def __getitem__(self, index):
        sample = {}
        sample['annotations'] = self._annotations[index]
        sample['file_name'] = self._image_path[index]
        sample['image_id'] = index
        sample = self._mapper(sample)
        return sample

class ImageDetectionDemoDataset(ImageDetectionDataset):
    def __init__(self):
        super().__init__(image_path=C.TEST_IMG_PATH,
                         annotations=[[{'bbox': [438, 254, 455, 271], 'bbox_mode': 0, 'category_id': 0},
                                       {'bbox': [388, 259, 408, 279], 'bbox_mode': 0, 'category_id': 1}]] * 2,
                         augmentations=[])

class ImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_path=None, labels=None, transforms=None):
        self._image_path = image_path
        self._labels = labels
        self._transforms = transforms

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        label = torch.tensor(np.float64(self._labels[index]))
        image = Image.open(self._image_path[index]).convert('RGB')
        if self._transforms is not None:
            image = self._transforms(image)

        return image, label

class ImageClassificationDemoDataset(ImageClassificationDataset):
    def __init__(self):
        super().__init__(image_path=C.TEST_IMG_PATH, labels=[
            0, 1], transforms=T.Compose([T.Resize((224, 224)), T.ToTensor()]))