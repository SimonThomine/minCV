import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
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
  
class SegmentationCustomDataset(Dataset):
    def __init__(self,path,image_size=(224,224),**kwargs):
        super().__init__()
        self.image_path=f"{path}/images"
        self.msk_path=f"{path}/annotations/trimaps"

        self.transform=transforms.Compose([transforms.Resize(image_size),transforms.ToTensor()])

        self.resize=transforms.Resize(image_size)

        self.data=self.get_data()

        test_img=self.transform(Image.open(self.data[0][0]))
        self.classes = len(np.unique(np.array(Image.open(self.data[0][1]).convert('L'))))
        
        self.image_dim = test_img.shape
        
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index) :
        img_path,label_path=self.data[index]
        img=Image.open(img_path)
        img=self.transform(img)
        # Handle grayscale images
        if img.shape[0] == 1:  
            img = img.repeat(3, 1, 1)
        # Handle RGBA images
        elif img.shape[0] == 4:
            img = img[:3]

        trimap=self.resize(Image.open(label_path).convert('L'))
        trimap=np.array(trimap)

        label = []
        for i in range(self.classes):
            class_mask = (trimap == i).astype(np.uint8)
            label.append(class_mask)
        
        label = np.stack(label, axis=0)
        label = torch.tensor(label)



        return img,label
      
    def get_data(self):
        data=[]

        for img in sorted(os.listdir(f"{self.image_path}")):
            if not img.lower().endswith((".jpg",".jpeg",".png")):
                continue
            img_path=f"{self.image_path}/{img}"
            label_path=f"{self.msk_path}/{os.path.splitext(img)[0]+'.png'}"
            data.append((img_path,label_path))
        return data
  

# import matplotlib.pyplot as plt
# dataset=SegmentationCustomDataset("data/Oxford-IIIT")
# img=dataset[0][0].permute(1,2,0).numpy()
# msk = dataset[0][1].permute(1, 2, 0).numpy() * 1.0

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# axs[0].imshow(img)
# axs[0].set_title('Image')
# axs[1].imshow(msk)
# axs[1].set_title('Masque')
# plt.show()