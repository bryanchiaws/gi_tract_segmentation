"""Define Logger class for logging information to stdout and disk."""
import json
import os
from os.path import join
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def get_ckpt_dir(save_path, exp_name):
    return os.path.join(save_path, exp_name, "ckpts")


def get_ckpt_callback(save_path, exp_name):
    ckpt_dir = os.path.join(save_path, exp_name, "ckpts")
    return ModelCheckpoint(dirpath=ckpt_dir,
                           save_top_k=1,
                           verbose=True,
                           monitor='val_loss',
                           mode='min')


def get_early_stop_callback(patience=10):
    return EarlyStopping(monitor='val_loss',
                         patience=patience,
                         verbose=True,
                         mode='min')


def get_logger(logger_type, save_path, exp_name, project_name=None):
    if logger_type == 'wandb':
        if project_name is None:
            raise ValueError("Must supply project name when using wandb logger.")
        return WandbLogger(name=exp_name,
                           project=project_name)
    elif logger_type == 'test_tube': 
        exp_dir = os.path.join(save_path, exp_name)
        return TestTubeLogger(save_dir=exp_dir,
                              name='lightning_logs',
                              version="0")
    else:
        raise ValueError(f'{logger_type} is not a supported logger.')

