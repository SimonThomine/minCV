
import torch
import torch.optim as optim
from trainers.base_trainer import BaseTrainer
from minMLP.mlp import Mlp
from torchvision import datasets, transforms



class MlpTrainer(BaseTrainer):
  def __init__(self, data):
    super().__init__(data)
    
    assert "input_dim" in self.data and isinstance(self.data["input_dim"],int), "input_dim not found in data or not an integer"
    assert "classes" in self.data and isinstance(self.data["classes"],int), "classes not found in data or not an integer"
    assert "hidden_dims" in self.data and all(isinstance(x, int) for x in self.data["hidden_dims"]), "hidden_dims not found in data or not a list of integers"
    assert "lr" in self.data and isinstance(self.data["lr"],float), "lr not found in data or not a float"
    
  def load_model(self):
    self.model=Mlp(**self.data)
    
  def load_optim(self):
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.data["lr"])
    
  def change_mode(self,period="train"):
    if period=="train":
      self.model.train()
    else:
      self.model.eval()



  # TODO A BOUGER POUR REFACTORISATION

  def load_data(self):
    # Load data with torchvision dataset
    
    
    if self.data["dataset"]=="mnist":
      trainval_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
      training_partition=int(len(trainval_dataset)*0.8)
      validation_partition=len(trainval_dataset)-training_partition
      train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
      test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    
    elif self.data["dataset"]=="cifar10":
      trainval_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
      training_partition=int(len(trainval_dataset)*0.8)
      validation_partition=len(trainval_dataset)-training_partition
      train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
      test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    
    elif self.data["dataset"]=="cifar100":
      trainval_dataset = datasets.CIFAR100(root='./data', train=True, transform=transforms.ToTensor(), download=True)
      training_partition=int(len(trainval_dataset)*0.8)
      validation_partition=len(trainval_dataset)-training_partition
      train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
      test_dataset = datasets.CIFAR100(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    
    elif self.data["dataset"]=="fashionmnist":
      trainval_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
      training_partition=int(len(trainval_dataset)*0.8)
      validation_partition=len(trainval_dataset)-training_partition
      train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
      test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    
    elif self.data["dataset"]=="kmnist":
      trainval_dataset = datasets.KMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
      training_partition=int(len(trainval_dataset)*0.8)
      validation_partition=len(trainval_dataset)-training_partition
      train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
      test_dataset = datasets.KMNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    
    elif self.data["dataset"]=="svhn":
      trainval_dataset = datasets.SVHN(root='./data', split='train', transform=transforms.ToTensor(), download=True)
      training_partition=int(len(trainval_dataset)*0.8)
      validation_partition=len(trainval_dataset)-training_partition
      train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
      test_dataset = datasets.SVHN(root='./data', split='test', transform=transforms.ToTensor(), download=True)
    
    elif self.data["dataset"]=="stl10":
      trainval_dataset = datasets.STL10(root='./data', split='train', transform=transforms.ToTensor(), download=True)
      training_partition=int(len(trainval_dataset)*0.8)
      validation_partition=len(trainval_dataset)-training_partition
      train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [training_partition, validation_partition])
      test_dataset = datasets.STL10(root='./data', split='test', transform=transforms.ToTensor(), download=True)
    
    
    self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.data["batch_size"], shuffle=True)
    self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.data["batch_size"], shuffle=False)
    self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.data["batch_size"], shuffle=False)
      
  