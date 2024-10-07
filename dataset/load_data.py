import torch
from torchvision import datasets, transforms
from dataset.custom_datasets import ClassificationCustomDataset,AutoencoderCustomDataset

class DatasetInfo():
    def __init__(self, dataset, input_dim):
        self.dataset = dataset
        self.input_dim = input_dim
    
    def __repr__(self):
        return f'dataset : {self.dataset}, input dimension : {self.input_dim}'

class ClassiDatasetInfo(DatasetInfo):
    def __init__(self, dataset, classes, input_dim):
        super().__init__(dataset, input_dim)
        self.classes = classes
    def __repr__(self) :
        return f'dataset : {self.dataset}, number of classes : {self.classes}, input dimension : {self.input_dim}'


dataset_mapping = {
    "mnist": ClassiDatasetInfo(datasets.MNIST, 10, 784),
    "cifar10": ClassiDatasetInfo(datasets.CIFAR10, 10, 3072),
    "cifar100": ClassiDatasetInfo(datasets.CIFAR100, 100, 3072),
    "fashionmnist": ClassiDatasetInfo(datasets.FashionMNIST, 10, 784),
    "kmnist": ClassiDatasetInfo(datasets.KMNIST, 10, 784),
    "svhn": ClassiDatasetInfo(datasets.SVHN, 10, 3072),
    "stl10": ClassiDatasetInfo(datasets.STL10, 10, 3072),
}

def load_classi_dataset(dataset,batch_size,image_size=(224,224),**kwargs):
    info=dataset_mapping.get(dataset)
    if info:
        train_dataset, val_dataset, test_dataset = load_known_dataset(info)
    else :
        print(f"Dataset {dataset} not in the default choices. Proceed with custom dataset")
        trainval_dataset=ClassificationCustomDataset(f"data/{dataset}","train",image_size=image_size)
        training_partition=int(len(trainval_dataset)*0.8)
        validation_partition=len(trainval_dataset)-training_partition
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
        
        test_dataset=ClassificationCustomDataset(f"data/{dataset}","val",image_size=image_size)

        info=ClassiDatasetInfo(dataset,len(trainval_dataset.classes),trainval_dataset.input_dim)

    print(info)
    train_loader, val_loader, test_loader = load_loaders(train_dataset, val_dataset, test_dataset,batch_size=batch_size)
    return train_loader, val_loader, test_loader,info.classes,info.input_dim

def load_autoencoder_dataset(dataset,batch_size,image_size=(224,224),**kwargs):
    info=dataset_mapping.get(dataset)
    if info:
        train_dataset, val_dataset, test_dataset = load_known_dataset(info)
    else :
        print(f"Dataset {dataset} not in the default choices. Proceed with custom dataset")
        trainval_dataset=AutoencoderCustomDataset(f"data/{dataset}","train",image_size=image_size)
        training_partition=int(len(trainval_dataset)*0.8)
        validation_partition=len(trainval_dataset)-training_partition
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
        
        test_dataset=AutoencoderCustomDataset(f"data/{dataset}","test",image_size=image_size)

        info=DatasetInfo(dataset,trainval_dataset.input_dim)

    print(info)
    train_loader, val_loader, test_loader = load_loaders(train_dataset, val_dataset, test_dataset,batch_size=batch_size)
    return train_loader, val_loader, test_loader,info.input_dim

def load_known_dataset(info):
    act_dataset=info.dataset
    trainval_dataset = act_dataset(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    training_partition=int(len(trainval_dataset)*0.8)
    validation_partition=len(trainval_dataset)-training_partition
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
    test_dataset = act_dataset(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    return train_dataset, val_dataset, test_dataset

def load_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader