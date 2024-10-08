import torch
import torch.nn as nn
import torch.nn.functional as F
from minBackbones.layers import BaseLayer,CnnLayer,MlpLayer, CnnLayerT

class Cnn(nn.Module):
  def __init__(self,image_dim, layers: list[BaseLayer],type="classification",**kwargs):
    super().__init__()

    self.convnet=nn.ModuleList()
    if type=="classification":
        self.mlphead=nn.ModuleList()
    if type=="autoencoder":
       self.decoder=nn.ModuleList()
    self.type=type
    
    # channel and height/width of the input
    previous_filt=image_dim[0]
    previous_size=(image_dim[1],image_dim[2])
    previous_dim=0
    for layer in  layers:
      if isinstance(layer, CnnLayer):
        self.add_layer(previous_filt, layer)
        previous_filt=layer.filters
        previous_size=((previous_size[0]-layer.kernel_size+2*layer.padding)//layer.stride+1, 
                       (previous_size[1]-layer.kernel_size+2*layer.padding)//layer.stride+1)
      elif isinstance(layer, MlpLayer):
        # cnn ends with a mlp
        if previous_dim==0:
            previous_dim=previous_filt*previous_size[0]*previous_size[1]
        self.add_layer(previous_dim, layer)
        previous_dim=layer.hidden_dim
      elif isinstance(layer, CnnLayerT):
        self.add_layer(previous_filt, layer)
        previous_filt=layer.filters
        previous_size=((previous_size[0]-1)*layer.stride+layer.kernel_size-2*layer.padding, 
                       (previous_size[1]-1)*layer.stride+layer.kernel_size-2*layer.padding)
    
    # Head depending on the type of task
    if type=="classification":
      # Softmax and sigmoid are handled in the loss function
      classes=kwargs.get("classes")
      self.mlphead.append(nn.Linear(previous_dim, classes if classes>2 else 1))
    # No mlp head for autoencoder
    elif type=="autoencoder":
      # Bof ça
      self.decoder.append(nn.ConvTranspose2d(previous_filt, image_dim[0], kernel_size=4, stride=2, padding=1))
    

  def add_layer(self, in_filters,layer):
    if isinstance(layer, CnnLayer):
        self.convnet.append(nn.Conv2d(in_filters, layer.filters, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding))
        self.convnet.append(layer.act)
        if layer.bn:
            self.convnet.append(nn.BatchNorm2d(layer.filters))
        self.convnet.append(nn.Dropout(p=layer.dropout))
    elif isinstance(layer, MlpLayer):
        self.mlphead.append(nn.Linear(in_filters, layer.hidden_dim))
        self.mlphead.append(layer.act)
        if layer.bn:
            self.mlphead.append(nn.BatchNorm1d(layer.hidden_dim))
        
        self.mlphead.append(nn.Dropout(p=layer.dropout))
    elif isinstance(layer, CnnLayerT):
        self.decoder.append(nn.ConvTranspose2d(in_filters, layer.filters, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding))
        self.decoder.append(layer.act)
        if layer.bn:
            self.decoder.append(nn.BatchNorm2d(layer.filters))
        self.decoder.append(nn.Dropout(p=layer.dropout))
  
  
  def forward(self, x):
    for layer in self.convnet:
      x=layer(x)
    # Flatten the output of the convnet
    if self.type=="classification":
        x=x.view(x.size(0), -1)
        for layer in self.mlphead:
            x=layer(x)
        return x.squeeze()
    elif self.type=="autoencoder":
       for layer in self.decoder:
        x=layer(x)
       return x  
    
     