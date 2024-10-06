from trainers.mlp_trainer import MlpTrainer


# classification datasets : mnist, cifar10, cifar100, fashionmnist, kmnist, svhn, stl10
conf={
    "hidden_dims":[256,256,128,128,64],
    "lr":0.001,
    "num_epochs":20,
    "batch_size":64,
    "dataset":"hymenoptera",
    "device":"mps",
    "image_size":(224,224) # (224,224) if not specified
}

trainer=MlpTrainer(conf)
trainer.train()
trainer.test()