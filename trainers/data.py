import os
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class DatasetInfo():
    def __init__(self, dataset, classes, input_dim):
        self.dataset = dataset
        self.classes = classes
        self.input_dim = input_dim
    def __repr__(self) :
        return f'dataset : {self.dataset}, number of classes : {self.classes}, input dimension : {self.input_dim}'


class ClassificationCustomDataset(Dataset):
  def __init__(self,path,split="train"):
      super().__init__()
      self.path=path
      self.split=split
      
      # Create corresponding dict for labels
      self.label_dict={}
      for i,label in enumerate(os.listdir(path+"/train")):
          self.label_dict[label]=i
          
      self.data=self.get_data()
      
  def __len__(self):
      return len(self.data)
  
  def __getitem__(self, index) :
      img_path,label=self.data[index]
      img=Image.open(img_path)
      img=transforms.ToTensor()(img)
      return img,label
      
  def get_data(self):
      # Train data
      path=f"{self.path}/{self.split}"
      data=[]
      for label in os.listdir(path):
          for img in os.listdir(path+"/"+label):
              img_path=path+"/"+label+"/"+img
              data.append((img_path,self.label_dict[label]))
      return data
    
    
  def __len__(self):
      return len(self.train_data)


dataset_mapping = {
    "mnist": DatasetInfo(datasets.MNIST, 10, 784),
    "cifar10": DatasetInfo(datasets.CIFAR10, 10, 3072),
    "cifar100": DatasetInfo(datasets.CIFAR100, 100, 3072),
    "fashionmnist": DatasetInfo(datasets.FashionMNIST, 10, 784),
    "kmnist": DatasetInfo(datasets.KMNIST, 10, 784),
    "svhn": DatasetInfo(datasets.SVHN, 10, 3072),
    "stl10": DatasetInfo(datasets.STL10, 10, 3072),
}

def load_dataset(name,batch_size,type="classification"):
    info=dataset_mapping.get(name)
    print(info)
    act_dataset=info.dataset
    
    if act_dataset is None:
        raise ValueError(f"Dataset {name} not found.")
      
    trainval_dataset = act_dataset(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    training_partition=int(len(trainval_dataset)*0.8)
    validation_partition=len(trainval_dataset)-training_partition
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
    test_dataset = act_dataset(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,val_loader,test_loader,info.classes,info.input_dim