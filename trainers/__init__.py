from .base_trainer import BaseTrainer
from .autoencoder_trainer import AeTrainer
from .classification_trainer import ClassiTrainer
from .segmentation_trainer import SegTrainer

TRAINERS={
    "classification":ClassiTrainer,
    "autoencoder":AeTrainer,
    "segmentation":SegTrainer,
}

__all__ = ['BaseTrainer', 'AeTrainer', 'ClassiTrainer','SegTrainer','TRAINERS']