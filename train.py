from trainers import TRAINERS
from minBackbones import CnnLayer,MlpLayer,CnnLayerT,TransformerParams


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


# conf={
#     "model_family":"mlp",
#     "layers":[MlpLayer(256),MlpLayer(256),MlpLayer(128),MlpLayer(128),MlpLayer(64)],
#     "lr":0.001,
#     "num_epochs":20,
#     "batch_size":64,
#     "dataset":"hymenoptera", #hymenoptera  mvtec/carpet
#     "type":"classification",
#     "device":"mps",
#     "image_size":(224,224) # (224,224) if not specified
# }

# TODO default model family
# conf={
#     "model_family":"cnn",
#     "layers": [CnnLayer(32),CnnLayer(64),CnnLayer(128),CnnLayer(256),MlpLayer(512),MlpLayer(256)],
#     "lr":0.001,
#     "num_epochs":20,
#     "batch_size":64,
#     "dataset":"hymenoptera", #hymenoptera  mvtec/carpet
#     "type":"classification",
#     "device":"mps",
#     "image_size":(224,224) # (224,224) if not specified
# }

conf={
    "model_family":"cnn",
    "layers": TransformerParams(patch_size=32,n_embd=64,n_head=8,n_layers=6),
    "lr":0.001,
    "num_epochs":20,
    "batch_size":64,
    "dataset":"hymenoptera", #hymenoptera  mvtec/carpet
    "type":"classification",
    "device":"mps",
    "image_size":(224,224) # (224,224) if not specified
}


trainer_type=TRAINERS[conf["type"]]
trainer=trainer_type(conf)

trainer.train()
trainer.test()