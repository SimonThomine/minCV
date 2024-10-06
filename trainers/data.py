import os
import torch
import numpy as np
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
  def __init__(self,path,split="train",image_size=(224,224),**kwargs):
      super().__init__()
      self.path=f"{path}/{split}"
      
      # Create corresponding dict for labels
      self.label_dict={}

    
      self.classes=[item for item in os.listdir(self.path) if os.path.isdir(f"{self.path}/{item}")]

      for i,label in enumerate(self.classes):
          self.label_dict[label]=torch.tensor(float(i))

      self.transform=transforms.Compose([transforms.Resize(image_size),transforms.ToTensor()])

      self.data=self.get_data()


      test_img=self.transform(Image.open(self.data[0][0]))
      if (shape := test_img.shape) and len(shape) == 3:
        self.input_dim = shape[0] * shape[1] * shape[2]
      else:
        self.input_dim = shape[0] * shape[1]
      
  def __len__(self):
      return len(self.data)
  
  def __getitem__(self, index) :
      img_path,label=self.data[index]
      img=Image.open(img_path)
      img=self.transform(img)
      return img,label
      
  def get_data(self):
      data=[]
      for label in self.classes:
          for img in os.listdir(f"{self.path}/{label}"):
              if not img.lower().endswith((".jpg",".jpeg",".png")):
                  continue
              img_path=f"{self.path}/{label}/{img}"
              data.append((img_path,self.label_dict[label]))
      return data


dataset_mapping = {
    "mnist": DatasetInfo(datasets.MNIST, 10, 784),
    "cifar10": DatasetInfo(datasets.CIFAR10, 10, 3072),
    "cifar100": DatasetInfo(datasets.CIFAR100, 100, 3072),
    "fashionmnist": DatasetInfo(datasets.FashionMNIST, 10, 784),
    "kmnist": DatasetInfo(datasets.KMNIST, 10, 784),
    "svhn": DatasetInfo(datasets.SVHN, 10, 3072),
    "stl10": DatasetInfo(datasets.STL10, 10, 3072),
}

def load_dataset(dataset,batch_size,type="classification",image_size=(224,224),**kwargs):
    info=dataset_mapping.get(dataset)
    if info:
        act_dataset=info.dataset
        trainval_dataset = act_dataset(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        training_partition=int(len(trainval_dataset)*0.8)
        validation_partition=len(trainval_dataset)-training_partition
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
        test_dataset = act_dataset(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    else :
        print(f"Dataset {dataset} not in the default choices. Proceed with custom dataset")
        trainval_dataset=ClassificationCustomDataset(f"data/{dataset}","train",image_size=image_size)
        training_partition=int(len(trainval_dataset)*0.8)
        validation_partition=len(trainval_dataset)-training_partition
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
        
        test_dataset=ClassificationCustomDataset(f"data/{dataset}","val",image_size=image_size)

        info=DatasetInfo(dataset,len(trainval_dataset.classes),trainval_dataset.input_dim)

    print(info)

    
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,val_loader,test_loader,info.classes,info.input_dim