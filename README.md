# Starter
[![Build Status](https://circleci.com/gh/stanfordmlgroup/starter.svg?style=svg&circle-token=221f645526d477538b83dfe6b360cb9941812d67)](https://circleci.com/gh/stanfordmlgroup/starter)
[![CodeFactor](https://www.codefactor.io/repository/github/stanfordmlgroup/starter/badge?s=40e8dc5c5da01117c4b8999f9f8326c5cd3bdf40)](https://www.codefactor.io/repository/github/stanfordmlgroup/starter)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>
This is a starter repo providing skeleton of ML projects for image classification, image segmentation and image generation. Please fork this repo to get started and share your brilliant ideas through pull request. 


## Quick start
### Set up virtual environment
`conda env create -f environment.yml`

`conda activate cs231n`

`pip install -r requirements.txt`

### Download Kaggle datasets
`pip install kaggle`

`Follow instructions here to create API token: https://github.com/Kaggle/kaggle-api#api-credentials`

`kaggle competitions download -c uw-madison-gi-tract-image-segmentation`

### Unzip dataset once you have it installed. The dataset should be in a folder called train
`tar -xvzf uw-madison-gi-tract-image-segmentation.zip`

[Optional] Rename dataset folder to something more intuitive

`import os`

`os.rename("train", "datasets")`

### Train and save a model 
`python main.py train --<hyperparameter> value`

### Test existing model 
`python main.py test --checkpoint_path <path to checkpoint>`

## Repo structure
This repo is designed to speed up th research iteration in the early stage of the project. 
Some design principles we followed: 
- Centralize the logic of configuration
- Include only necessary kick-starter pieces 
- Only abstract the common component and structure across projects
- Expose 100% data loading logic, model architecture and forward/backward logic in original PyTorch
- Be prepared to hyper-changes

### What you might want to modify and where are they?
#### Main configuration
`main.py` defines all the experiments level configuration (e.g. which model/optimizer to use, how to decay the learning rate, when to save the model and where, and etc.). We use [Fire](https://github.com/google/python-fire/blob/master/docs/guide.md) to automatically generate CLI for function like `train(...)` and `test(...)`. For most of the hyper-parameter searching experiments, modifying `main.py` should be enough

To further modify the training loop logic (for GAN, meta-learning, and etc.), you may want to update the `train(...)` and `test(...)` functions. You can try all your crazy research ideas there!

#### Dataset 
`data/dataset.py` provides a basic example but you probably want to define your own dataset with on-the-fly transforms and augmentations. This can be done by implement your class of dataset and transforming functions in `data` module and use them in `train/valid/test_dataloader()` in `lightning/model.py`. If you have a lot of dataset, you might also want to implement some `get_dataset(args)` method to help fetch the correct dataset. 

#### Model architecture
We include most of the established backbone models in `models/pretrained.py` but you are welcome to implement your own, just as what you did in plain PyTorch. 

#### Others
We would suggest you to put the implementation of optimizer, loss, evaluation metrics, logger and constants into `/util`. 

For other project-specified codes (such as pre-processing and data visualization), you might want to leave them to `/custom`.

## Useful links 
- [Negative sampling (Google Map API)](https://github.com/stanfordmlgroup/old-starter/blob/master/preprocess/get_negatives.py)
- [Example of dataset implementation: USGS dataset](https://github.com/stanfordmlgroup/old-starter/blob/master/data/usgs_dataset.py)
- [Documentation for Fire](https://github.com/google/python-fire/blob/master/docs/guide.md)
- [Documentation for Pytorch Lighning](https://pytorch-lightning.readthedocs.io/en/stable/)


## Troubleshooting Notes
- Inplace operations in PyTorch is not supported in PyTorch Lightning distributed mode. Please just use non-inplace operations instead. 

--- 
Maintainers: [@Hao](mailto:haosheng@cs.stanford.edu)
 
