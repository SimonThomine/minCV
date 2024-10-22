import torch.nn as nn
import timm


# TODO ajouter option fine tuning etc

class Tim(nn.Module):
  def __init__(self, image_dim, layers: str,type="classification",pretrained=True,**kwargs):
    super().__init__()
  
    self.net = timm.create_model(layers, pretrained=pretrained)

    #Freeze the layers
    for param in self.net.parameters():
      param.requires_grad = False

    if list(image_dim) != [3,224,224]:
      raise ValueError("Timm only accept 224x224x3 images (for now)")

    self.type=type

    # Head depending on the type of task
    if type=="classification":
      # Softmax and sigmoid are handled in the loss function
      classes=kwargs.get("classes")
      num_features=self.net.fc.in_features
      self.net.fc=nn.Linear(num_features, classes if classes>2 else 1)
      # Unfreeze the last layer
      self.net.fc.requires_grad=True
    else:
      raise NotImplementedError("Only classification is implemented for now")
    
  def forward(self, x):
    return self.net(x).squeeze()
  