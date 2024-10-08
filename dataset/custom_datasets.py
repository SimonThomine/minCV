import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class ClassificationCustomDataset(Dataset):
  def __init__(self,path,split="train",image_size=(224,224),**kwargs):
      super().__init__()
      self.path=f"{path}/{split}"
      
      self.label_dict={}

    
      self.classes=[item for item in os.listdir(self.path) if os.path.isdir(f"{self.path}/{item}")]

      for i,label in enumerate(self.classes):
          self.label_dict[label]=torch.tensor(float(i))

      self.transform=transforms.Compose([transforms.Resize(image_size),transforms.ToTensor()])

      self.data=self.get_data()


      test_img=self.transform(Image.open(self.data[0][0]))

      self.image_dim = test_img.shape
      
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
  
class AutoencoderCustomDataset(Dataset):
  def __init__(self,path,split="train",image_size=(224,224),**kwargs):
      super().__init__()
      self.path=f"{path}/{split}"

      self.transform=transforms.Compose([transforms.Resize(image_size),transforms.ToTensor()])

      self.data=self.get_data()


      test_img=self.transform(Image.open(self.data[0][0]))
      
      self.image_dim = test_img.shape
      
  def __len__(self):
      return len(self.data)
  
  def __getitem__(self, index) :
      img_path,label=self.data[index]
      img=Image.open(img_path)
      img=self.transform(img)
      return img,label
      
  def get_data(self):
    data=[]
    categories=os.listdir(self.path)
    for cat in categories:
        if cat=="good":
            label=torch.tensor(0)
        else:
            label=torch.tensor(1)
        for img in os.listdir(f"{self.path}/{cat}"):
            if not img.lower().endswith((".jpg",".jpeg",".png")):
                continue
            img_path=f"{self.path}/{cat}/{img}"
            data.append((img_path,label))
    return data