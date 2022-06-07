import os
import fire
from pytorch_lightning import Trainer
import numpy as np
import pdb

from util import init_exp_folder, Args
from lightning import (get_task,
                       load_task,
                       get_ckpt_callback, 
                       get_early_stop_callback,
                       get_logger)


def train(dataset_folder="./data_files",
          save_dir="./sandbox",
          exp_name="DemoExperiment",
          model="ResNet18",
          task='classification',
          gpus=1,
          pretrained=True,
          num_classes=1,
          accelerator=None,
          logger_type='test_tube',
          gradient_clip_val=0.5,
          max_epochs=1,
          patience=10,
          stochastic_weight_avg=True,
          limit_train_batches=1.0,
          tb_path="./sandbox/tb",
          loss_fn="BCE",
          weights_summary=None,
          augmentation = 'none',
          num_workers=0,
          auto_lr_find= True,
          lr = 0.001,
          batch_size = 2,
 #         pretraining = False,
 #         aux_task = None
          ):
    """
    Run the training experiment.

    Args:
        save_dir: Path to save the checkpoints and logs
        exp_name: Name of the experiment
        model: Model name
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
        pretrained: Whether or not to use the pretrained model
        num_classes: Number of classes
        accelerator: Distributed computing mode
        logger_type: 'wandb' or 'test_tube'
        gradient_clip_val:  Clip value of gradient norm
        limit_train_batches: Proportion of training data to use
        max_epochs: Max number of epochs
        patience: number of epochs with no improvement after
                  which training will be stopped.
        stochastic_weight_avg: Whether to use stochastic weight averaging.
        tb_path: Path to global tb folder
        loss_fn: Loss function to use
        weights_summary: Prints a summary of the weights when training begins.

    Returns: None

    """
    args = Args(locals()) #Allows you to access stuff in the dictionary as args.exp_name
    init_exp_folder(args) #Sets up experiment directory 
    task = get_task(args) #Have to define this pytorch lightning module, for implementation, where constructor in segmentation.py is called
    #Then you instantiate trainer and start training
    trainer = Trainer(gpus=gpus, 
                      accelerator=accelerator,
                      logger=get_logger(logger_type, save_dir, exp_name), #Logging tool
                      callbacks=[get_early_stop_callback(patience), #Set number of epochs without improvement before stopping
                                 get_ckpt_callback(save_dir, exp_name)], #Save model checkpoints to folder, defined by certain metrics
                      weights_save_path=os.path.join(save_dir, exp_name), 
                      gradient_clip_val=None,
                      limit_train_batches=limit_train_batches, #When debugging, limit number of batches to run (percentage of data)
                      weights_summary=weights_summary,
                      stochastic_weight_avg=stochastic_weight_avg,
                      max_epochs=max_epochs,
                      auto_lr_find=True,
                      reload_dataloaders_every_n_epochs=1,
                      log_every_n_steps=1) #Handles functionality of training
    trainer.fit(task)


def test(ckpt_path,
         gpus=0,
         **kwargs):
    """
    Run the testing experiment.

    Args:
        ckpt_path: Path for the experiment to load
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
    Returns: None

    """
    task = load_task(ckpt_path, **kwargs)
    trainer = Trainer(gpus=gpus)
    trainer.test(task)

def predict(ckpt_path, gpus=1, prediction_path="predictions.pt", **kwargs):
    # couldn't figure out how to pass in a specific dataset as an argument
    # by default, this makes predictions over the test dataset
    # can change the prediction dataset in predict_dataloader() function in segmentation.py
    task = load_task(ckpt_path, **kwargs)
    trainer = Trainer(gpus=gpus)
    trainer.predict(task)
    preds_tensor = task.evaluator.preds
    preds = preds_tensor.cpu().detach()
    torch.save(prediction_path, predictions)

if __name__ == "__main__":
    fire.Fire() #Allows you to run functions and supply arguments directly in command line
