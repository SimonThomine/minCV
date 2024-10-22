import torch
from torchvision import datasets, transforms
from dataset.custom_datasets import ClassificationCustomDataset,AutoencoderCustomDataset,SegmentationCustomDataset
from medmnist import ChestMNIST,BloodMNIST,PathMNIST,ChestMNIST,PneumoniaMNIST,RetinaMNIST,OCTMNIST,DermaMNIST,BreastMNIST,TissueMNIST,OrganAMNIST,OrganCMNIST,OrganSMNIST


class DatasetInfo():
    def __init__(self, dataset, image_dim,origin=None):
        self.dataset = dataset
        self.image_dim = image_dim
        self.origin=origin
    
    def __repr__(self):
        return f'dataset : {self.dataset}, input dimension : {self.image_dim}'

class ClassiDatasetInfo(DatasetInfo):
    def __init__(self, dataset, classes, image_dim,origin=None):
        super().__init__(dataset, image_dim,origin)
        self.classes = classes
    def __repr__(self) :
        return f'dataset : {self.dataset}, number of classes : {self.classes}, input dimension : {self.image_dim}, origin : {self.origin}'
    
# Basically the same as ClassiDatasetInfo
class SegDatasetInfo(DatasetInfo):
    def __init__(self, dataset, classes, image_dim):
        super().__init__(dataset, image_dim)
        self.classes = classes
    def __repr__(self) :
        return f'dataset : {self.dataset}, number of classes : {self.classes}, input dimension : {self.image_dim}'


dataset_mapping = {
    "mnist": ClassiDatasetInfo(datasets.MNIST, 10, (1,28,28),origin="torchvision"),
    "cifar10": ClassiDatasetInfo(datasets.CIFAR10, 10, (3,32,32),origin="torchvision"),
    "cifar100": ClassiDatasetInfo(datasets.CIFAR100, 100, (3,32,32),origin="torchvision"),
    "fashionmnist": ClassiDatasetInfo(datasets.FashionMNIST, 10, (1,28,28),origin="torchvision"),
    "kmnist": ClassiDatasetInfo(datasets.KMNIST, 10, (1,28,28),origin="torchvision"),
    "chestmnist": ClassiDatasetInfo(ChestMNIST, 14, (1,28,28),origin="medmnist"),
    "bloodmnist": ClassiDatasetInfo(BloodMNIST, 8, (3,28,28),origin="medmnist"),
    "pathmnist": ClassiDatasetInfo(PathMNIST, 9, (3,28,28),origin="medmnist"),
    "pneumoniamnist": ClassiDatasetInfo(PneumoniaMNIST, 2, (1,28,28),origin="medmnist"),
    "octmnist": ClassiDatasetInfo(OCTMNIST, 4, (1,28,28),origin="medmnist"),
    "dermamnist": ClassiDatasetInfo(DermaMNIST, 7, (3,28,28),origin="medmnist"),
    "retinamnist": ClassiDatasetInfo(RetinaMNIST, 5, (3,28,28),origin="medmnist"),
    "breastmnist": ClassiDatasetInfo(BreastMNIST, 2, (1,28,28),origin="medmnist"),
    "tissuemnist": ClassiDatasetInfo(TissueMNIST, 8, (1,28,28),origin="medmnist"),
    "organamnist": ClassiDatasetInfo(OrganAMNIST, 11, (1,28,28),origin="medmnist"),
    "organcmnist": ClassiDatasetInfo(OrganCMNIST, 11, (1,28,28),origin="medmnist"),
    "organsmnist": ClassiDatasetInfo(OrganSMNIST, 11, (1,28,28),origin="medmnist"),
}

def load_classi_dataset(dataset,batch_size,image_size=(224,224),**kwargs):
    info=dataset_mapping.get(dataset)
    if info:
        train_dataset, val_dataset, test_dataset = load_known_dataset(info,image_size)
    else :
        print(f"Dataset {dataset} not in the default choices. Proceed with custom dataset")

        try:
            trainval_dataset = ClassificationCustomDataset(f"data/{dataset}", "train", image_size=image_size)
            train_dataset, val_dataset = split_dataset_train_val(trainval_dataset)
            test_dataset = ClassificationCustomDataset(f"data/{dataset}", "val", image_size=image_size)
        except FileNotFoundError:
            trainval_dataset = ClassificationCustomDataset(f"data/{dataset}", None, image_size=image_size)
            train_dataset, val_dataset, test_dataset = split_dataset_train_val_test(trainval_dataset)


        info=ClassiDatasetInfo(dataset,len(trainval_dataset.classes),trainval_dataset.image_dim)

    print(info)
    train_loader, val_loader, test_loader = load_loaders(train_dataset, val_dataset, test_dataset,batch_size=batch_size)
    return train_loader, val_loader, test_loader,info

def load_autoencoder_dataset(dataset,batch_size,image_size=(224,224),**kwargs):
    info=dataset_mapping.get(dataset)
    if info:
        train_dataset, val_dataset, test_dataset = load_known_dataset(info)
    else :
        print(f"Dataset {dataset} not in the default choices. Proceed with custom dataset")
        trainval_dataset=AutoencoderCustomDataset(f"data/{dataset}","train",image_size=image_size)
        train_dataset, val_dataset = split_dataset_train_val(trainval_dataset)
        test_dataset=AutoencoderCustomDataset(f"data/{dataset}","test",image_size=image_size)

        info=DatasetInfo(dataset,trainval_dataset.image_dim)

    print(info)
    train_loader, val_loader, test_loader = load_loaders(train_dataset, val_dataset, test_dataset,batch_size=batch_size)
    return train_loader, val_loader, test_loader,info

def load_seg_dataset(dataset,batch_size,image_size=(224,224),**kwargs):
    info=dataset_mapping.get(dataset)

    complete_dataset=SegmentationCustomDataset(f"data/{dataset}",image_size=image_size)
    train_dataset, val_dataset,test_dataset = split_dataset_train_val_test(complete_dataset)

    info=SegDatasetInfo(dataset,complete_dataset.classes,complete_dataset.image_dim)

    train_loader, val_loader, test_loader = load_loaders(train_dataset, val_dataset, test_dataset,batch_size=batch_size)
    return train_loader, val_loader, test_loader,info



def load_known_dataset(info,image_size):

    act_dataset=info.dataset
    if info.origin=="torchvision":
        trainval_dataset = act_dataset(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        train_dataset, val_dataset = split_dataset_train_val(trainval_dataset)
        test_dataset = act_dataset(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    elif info.origin=="medmnist":
        if info.image_dim[1]!=image_size[0]:
            print(f"Resizing image from {info.image_dim[1]} to {image_size[0]}")
            
            info.image_dim=(info.image_dim[0],image_size[0],image_size[1])
            print(f'info.image_dim={info.image_dim}') 
            
        transform=transforms.Compose([transforms.Resize((info.image_dim[1],info.image_dim[2])),transforms.ToTensor()])
        train_dataset = act_dataset(split='train', transform=transform, download=True)
        val_dataset = act_dataset(split='val', transform=transform, download=True)
        test_dataset = act_dataset(split='test', transform=transform, download=True)

    return train_dataset, val_dataset, test_dataset

def load_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def split_dataset_train_val(dataset,train_size=0.8):
    train_size = int(train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def split_dataset_train_val_test(dataset,train_size=0.8):
    train_size = int(train_size * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset 