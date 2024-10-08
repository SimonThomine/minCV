from .base_trainer import BaseTrainer
from .autoencoder_trainer import AeTrainer
from .classification_trainer import ClassiTrainer

TRAINERS={
    "classification":ClassiTrainer,
    "autoencoder":AeTrainer
}

__all__ = ['BaseTrainer', 'AeTrainer', 'ClassiTrainer']