import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
  def __init__(self, input_dim: int, classes: int, hidden_dims: list[int], act= nn.ReLU(),bn=False, dropout=False, dropout_p=0.5,**kwargs):
    super().__init__()
    self.net=nn.ModuleList()
    
    hidden_dims=[input_dim]+hidden_dims
    
    for in_feat, out_feat in  zip(hidden_dims,hidden_dims[1:]):
      self.add_layer(in_feat, out_feat, act=act, bn=bn, dropout=dropout, dropout_p=dropout_p)
     
    # Softmax and sigmoid are handled in the loss function
    self.net.append(nn.Linear(hidden_dims[-1], classes if classes>2 else 1))

      
  def add_layer(self, in_feat, out_feat,act,bn=False,dropout=False,dropout_p=0.5):
    self.net.append(nn.Linear(in_feat, out_feat))
    self.net.append(act)
    if bn:
      self.net.append(nn.BatchNorm1d(out_feat))
    if dropout:
      self.net.append(nn.Dropout(p=dropout_p))
  
  
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
     