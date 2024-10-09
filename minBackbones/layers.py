import torch.nn as nn

class BaseLayer:
    def __init__(self,act=nn.ReLU(),bn=True,dropout=0.):
        self.act=act
        self.bn=bn
        self.dropout=dropout

class MlpLayer(BaseLayer):
    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim=hidden_dim

class CnnLayer(BaseLayer):
    def __init__(self, filters: int,stride=2,kernel_size=3,padding=1,**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.stride=stride
        self.kernel_size=kernel_size
        self.padding=padding

class CnnLayerT(BaseLayer):
    def __init__(self, filters: int,stride=2,kernel_size=4,padding=1,**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.stride=stride
        self.kernel_size=kernel_size
        self.padding=padding

class ViTParams():
    def __init__(self, patch_size:int ,n_embd: int, n_head: int, n_layers: int,dropout=0.):
        self.n_embd=n_embd
        self.n_head=n_head
        self.n_layers=n_layers
        self.patch_size=patch_size
        self.dropout=dropout
    
