import torch.nn as nn
from minBackbones.layers import BaseLayer

class Mlp(nn.Module):
  def __init__(self, input_dim: int, layers: list[BaseLayer],type="classification",**kwargs):
    super().__init__()
    self.net=nn.ModuleList()
    
    previous_dim=input_dim
    for layer in  layers:
      self.add_layer(previous_dim, layer)
      previous_dim=layer.hidden_dim
     
    # Head depending on the type of task
    if type=="classification":
      # Softmax and sigmoid are handled in the loss function
      classes=kwargs.get("classes")
      self.net.append(nn.Linear(previous_dim, classes if classes>2 else 1))
    elif type=="autoencoder":
      self.net.append(nn.Linear(previous_dim, input_dim))
    

      
  def add_layer(self, in_feat, layer):
    self.net.append(nn.Linear(in_feat, layer.hidden_dim))
    self.net.append(layer.act)
    if layer.bn:
      self.net.append(nn.BatchNorm1d(layer.hidden_dim))
    if layer.dropout:
      self.net.append(nn.Dropout(p=layer.dropout_p))
  
  
  def forward(self, x):
    # If it is an image flatten it
    if len(x.shape) > 2:
      x = x.view(x.size(0), -1)
    
    for layer in self.net:
      x=layer(x)
    return x.squeeze()
  
  
# model=Mlp(input_dim=10, classes=5, hidden_dims=[32, 64, 128])
# print(model)
# dummy_input=torch.randn(10)
# out=model(dummy_input)
# print(out)
# print(out.shape)
     