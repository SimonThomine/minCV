import torch.nn as nn
from minBackbones.layers import BaseLayer

class Mlp(nn.Module):
  def __init__(self, image_dim, layers: list[BaseLayer],type="classification",**kwargs):
    super().__init__()
    self.net=nn.ModuleList()
    self.image_dim=image_dim  
    input_dim=image_dim[0]*image_dim[1]*image_dim[2]

    previous_dim=input_dim
    for layer in  layers:
      self.net.append(layer.create_layer(previous_dim))
      previous_dim=layer.hidden_dim
     

    self.type=type
    # Head depending on the type of task
    if type=="classification":
      # Softmax and sigmoid are handled in the loss function
      classes=kwargs.get("classes")
      self.net.append(nn.Linear(previous_dim, classes if classes>2 else 1))
    elif type=="autoencoder":
      self.net.append(nn.Linear(previous_dim, input_dim))
    elif type=="segmentation":
      classes=kwargs.get("classes")
      self.net.append(nn.Linear(previous_dim, classes*image_dim[1]*image_dim[2]))
      
  def forward(self, x):
    # If it is an image flatten it
    if len(x.shape) > 2:
      x = x.view(x.size(0), -1)
    for layer in self.net:
      x=layer(x)
    if self.type=="autoencoder":
      return x.view(x.size(0),*self.image_dim)
    elif self.type=="segmentation":
      return x.view(x.size(0),-1,self.image_dim[1],self.image_dim[2])
    elif self.type=="classification":
      return x.squeeze()
  
     