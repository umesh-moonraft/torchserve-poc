import os

from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.config import get_cfg as _get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

from loss import ValidationLoss

import cv2


def get_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes):
    """
    Create a Detectron2 configuration object and set its attributes.

    Args:
        output_dir (str): The path to the output directory where the trained model and logs will be saved.
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size used during training.
        iterations (int): The maximum number of training iterations.
        checkpoint_period (int): The number of iterations between consecutive checkpoints.
        model (str): The name of the model to use, which should be one of the models available in Detectron2's model zoo.
        device (str): The device to use for training, which should be 'cpu' or 'cuda'.
        nmr_classes (int): The number of classes in the dataset.

    Returns:
        The Detectron2 configuration object.
    """
    cfg = _get_cfg()

    # Merge the model's default configuration file with the default Detectron2 configuration file.
    cfg.merge_from_file(model_zoo.get_config_file(model))

    # Set the training and validation datasets and exclude the test dataset.
    # cfg.DATASETS.TRAIN = ("curtain_train_dataset", "chair_train_dataset")
    cfg.DATASETS.TRAIN = ("chair_train_dataset")
    cfg.DATASETS.VAL = ("chair_valid_dataset")  
    cfg.DATASETS.TEST = ()

    # Set the device to use for training.
    if device in ['cpu']:
        cfg.MODEL.DEVICE = 'cpu'

    # Set the number of data loader workers.
    cfg.DATALOADER.NUM_WORKERS = 2

    # Set the model weights to the ones pre-trained on the COCO dataset.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    # Set the batch size used by the solver.
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    # Set the checkpoint period.
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period

    # Set the base learning rate.
    cfg.SOLVER.BASE_LR = learning_rate

    # Set the maximum number of training iterations.
    cfg.SOLVER.MAX_ITER = iterations

    # Set the learning rate scheduler steps to an empty list, which means the learning rate will not be decayed.
    cfg.SOLVER.STEPS = []

    # Set the batch size used by the ROI heads during training.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # Set the number of classes.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nmr_classes

    # Set the output directory.
    cfg.OUTPUT_DIR = output_dir

    return cfg

def register_coco_datasets(class_list_file):
  datasets = [
    # {
    #   "name": "curtain_train_dataset",
    #   "json_file": "/home/desktop/Documents/vera-ml-torchserve/mr-training/train-detectron2/training-dataset/curtain-dataset/annotations/all_train_instances.json",
    #   "image_root": "/home/desktop/Documents/vera-ml-torchserve/mr-training/train-detectron2/training-dataset/curtain-dataset/images/train",
    # },
    # {
    #   "name": "curtain_valid_dataset",
    #   "json_file": "/home/desktop/Documents/vera-ml-torchserve/mr-training/train-detectron2/training-dataset/curtain-dataset/annotations/all_valid_instances.json",
    #   "image_root": "/home/desktop/Documents/vera-ml-torchserve/mr-training/train-detectron2/training-dataset/curtain-dataset/images/valid",
    # },
    {
      "name": "chair_train_dataset",
      "json_file": "/home/desktop/Documents/vera-ml-torchserve/mr-training/train-detectron2/training-dataset/office-chair-dataset/train/instances/train_instances.json",
      "image_root": "/home/desktop/Documents/vera-ml-torchserve/mr-training/train-detectron2/training-dataset/office-chair-dataset/train/images",
    },
    {
      "name": "chair_valid_dataset",
      "json_file": "/home/desktop/Documents/vera-ml-torchserve/mr-training/train-detectron2/training-dataset/office-chair-dataset/valid/annotations/valid_instances.json",
      "image_root": "/home/desktop/Documents/vera-ml-torchserve/mr-training/train-detectron2/training-dataset/office-chair-dataset/valid/images",
    }
    # Add more datasets as needed
  ]

  for dataset in datasets:
    register_coco_instances(dataset["name"], {}, dataset["json_file"], dataset["image_root"])
	
  with open(class_list_file, 'r') as reader:
        classes_ = [l[:-1] for l in reader.readlines()]
  return len(classes_)

def train(output_dir, class_list_file, learning_rate, batch_size, iterations, checkpoint_period, device,
          model):
    """
    Train a Detectron2 model on a custom dataset.

    Args:
        output_dir (str): Path to the directory to save the trained model and output files.
        data_dir (str): Path to the directory containing the dataset.
        class_list_file (str): Path to the file containing the list of class names in the dataset.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        iterations (int): Maximum number of training iterations.
        checkpoint_period (int): Number of iterations after which to save a checkpoint of the model.
        device (str): Device to use for training (e.g., 'cpu' or 'cuda').
        model (str): Name of the model configuration to use. Must be a key in the Detectron2 model zoo.

    Returns:
        None
    """

    # Register the dataset and get the number of classes
    nmr_classes = register_coco_datasets(class_list_file)

    # Get the configuration for the model
    cfg = get_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes)

    # Create the output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Create the trainer object
    trainer = DefaultTrainer(cfg)

    # Create a custom validation loss object
    val_loss = ValidationLoss(cfg)

    # Register the custom validation loss object as a hook to the trainer
    trainer.register_hooks([val_loss])

    # Swap the positions of the evaluation and checkpointing hooks so that the validation loss is logged correctly
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    # Resume training from a checkpoint or load the initial model weights
    trainer.resume_or_load(resume=True)

    # Train the model
    trainer.train()
