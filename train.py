from trainers import TRAINERS
from minBackbones import CnnLayer,MlpLayer,CnnLayerT,ViTParams


# classification datasets : mnist, cifar10, cifar100, fashionmnist, kmnist

# Tasks : 
# - classification
# - autoencoder : anomaly, compression, denoising
# - segmentation
# - object detection
# - next token prediction


# Comment définir un réseau convolutifs, hidden_dims=[
# Faire une classe layer et des sous-classes par type de couche
# Calcul automatique des dimensions si pas spécifié (pour les couches mlp de fin par exemple) (déjà fait en fait)
default_conf = {
    "lr": 0.001,
    "num_epochs": 10,
    "batch_size": 32,
    "device": "cpu",
    "image_size": (28, 28)
}

# conf={
#     "model_family":"mlp",
#     "layers":[MlpLayer(256),MlpLayer(256),MlpLayer(128),MlpLayer(128),MlpLayer(64)],
#     "dataset":"hymenoptera", #hymenoptera  mvtec/carpet
#     "type":"classification",
# }


conf={
    "model_family":"cnn",
    "layers": [CnnLayer(32),CnnLayer(64),CnnLayer(128),CnnLayer(256),MlpLayer(512),MlpLayer(256)],
    "dataset":"bloodmnist", #hymenoptera  mvtec/carpet
    "type":"classification",
}

# conf={
#     "model_family":"vit",
#     "layers": ViTParams(patch_size=32,n_embd=64,n_head=8,n_layers=6),
#     "dataset":"hymenoptera", #hymenoptera  mvtec/carpet
#     "type":"classification"
#     }

# conf={
#     "model_family":"cnn",
#     "layers": [CnnLayer(64),CnnLayer(128),CnnLayer(256),CnnLayer(512),CnnLayerT(256),CnnLayerT(128),CnnLayerT(64)],
#     "dataset":"mvtec/carpet", #hymenoptera  mvtec/carpet Oxford-IIIT
#     "type":"autoencoder",
#     "image_size": (224,224) 
# }

default_conf.update(conf)

trainer_type=TRAINERS[default_conf["type"]]
trainer=trainer_type(default_conf)

trainer.train()
trainer.test()