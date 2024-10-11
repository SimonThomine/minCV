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
    if type=="autoencoder" or type=="segmentation":
       self.decoder=nn.ModuleList()
    self.type=type
    
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
        if previous_dim==0:
            previous_dim=previous_filt*previous_size[0]*previous_size[1]
        self.add_layer(previous_dim, layer)
        previous_dim=layer.hidden_dim
      elif isinstance(layer, CnnLayerT):
        self.add_layer(previous_filt, layer)
        previous_filt=layer.filters
        previous_size=((previous_size[0]-1)*layer.stride+layer.kernel_size-2*layer.padding, 
                       (previous_size[1]-1)*layer.stride+layer.kernel_size-2*layer.padding)
    
    if type=="classification":
      # Softmax and sigmoid are handled in the loss function
      classes=kwargs.get("classes")
      self.mlphead.append(nn.Linear(previous_dim, classes if classes>2 else 1))

    elif type=="autoencoder" or type=="segmentation":
      self.decoder.append(nn.ConvTranspose2d(previous_filt, image_dim[0], kernel_size=4, stride=2, padding=1))
    
    elif type=="segmentation":
        classes=kwargs.get("classes")
        self.decoder.append(nn.ConvTranspose2d(previous_filt, classes, kernel_size=4, stride=2, padding=1))

    # TODO verif network, verify if dimensions are correct for the layer agencement
    

  def add_layer(self, in_filters,layer):
    if isinstance(layer, CnnLayer):
        self.convnet.append(layer.create_layer(in_filters))
    elif isinstance(layer, MlpLayer):
        self.mlphead.append(layer.create_layer(in_filters))
    elif isinstance(layer, CnnLayerT):
        self.decoder.append(layer.create_layer(in_filters))
  

  def forward_classi(self, x):
    for layer in self.convnet:
      x=layer(x)
    x=x.view(x.size(0), -1)
    for layer in self.mlphead:
      x=layer(x)
    return x.squeeze()
  
  def forward_ae(self, x):
    for layer in self.convnet:
      x=layer(x)
    for layer in self.decoder:
      x= layer(x)
    return x
  
  def forward_seg(self, x):
    intermediate=[]
    for layer in self.convnet:
      x=layer(x)
      if isinstance(layer, nn.Dropout):
        intermediate.append(x)
    # pop the last intermediate feature map
    x=intermediate.pop()
    for layer in self.decoder:
      x= intermediate.pop() + layer(x) if isinstance(layer,nn.Dropout) else layer(x)
    return x

  
  def forward(self, x):
    if self.type=="classification":
        return self.forward_classi(x)
    elif self.type=="autoencoder":
        return self.forward_ae(x)
    elif self.type=="segmentation":
        return self.forward_seg(x)
    else:
        raise ValueError("type not supported")
    
     