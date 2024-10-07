from trainers.classification_trainer import ClassiTrainer
from trainers.autoencoder_trainer import AeTrainer
from minBackbones.layers import BaseLayer,MlpLayer,CnnLayer

# classification datasets : mnist, cifar10, cifar100, fashionmnist, kmnist, svhn, stl10

# Tasks : 
# - classification
# - autoencoder : anomaly, compression, denoising
# - segmentation
# - object detection
# - next token prediction


# Comment définir un réseau convolutifs, hidden_dims=[
# Faire une classe layer et des sous-classes par type de couche
# Calcul automatique des dimensions si pas spécifié (pour les couches mlp de fin par exemple) (déjà fait en fait)


conf={
    "model_family":"mlp",
    "layers":[MlpLayer(256),MlpLayer(256),MlpLayer(128),MlpLayer(128),MlpLayer(64)],
    "lr":0.001,
    "num_epochs":5,
    "batch_size":64,
    "dataset":"mvtec/carpet", #hymenoptera  mvtec/carpet
    "type":"autoencoder",
    "device":"mps",
    "image_size":(224,224) # (224,224) if not specified
}

if conf["type"]=="classification":
    trainer=ClassiTrainer(conf)
elif conf["type"]=="autoencoder":
    trainer=AeTrainer(conf)

trainer.train()
trainer.test()