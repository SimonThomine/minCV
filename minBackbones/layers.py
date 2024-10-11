import torch.nn as nn

NORM_DICT = {
    'batchnorm2d': lambda num_features: nn.BatchNorm2d(num_features),
    'batchnorm1d': lambda num_features: nn.BatchNorm1d(num_features),
    'layernorm': lambda num_features: nn.LayerNorm(num_features),
    'groupnorm': lambda num_features: nn.GroupNorm(num_features//2, num_features),
    'instancenorm2d': lambda num_features: nn.InstanceNorm2d(num_features),
    'instancenorm1d': lambda num_features: nn.InstanceNorm1d(num_features)
}

class BaseLayer:
    def __init__(self,act=nn.ReLU(),norm='batchnorm2d',dropout=0.):
        self.act=act
        self.norm=norm
        self.dropout=dropout

class MlpLayer(BaseLayer):
    def __init__(self, hidden_dim: int,norm='batchnorm1d', **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim=hidden_dim
        self.norm=norm
    
    def create_layer(self,in_dim,**kwargs):
        layer=nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            self.act,
            NORM_DICT[self.norm](self.hidden_dim),
            nn.Dropout(p=self.dropout)
        )
        return layer

class CnnLayer(BaseLayer):
    def __init__(self, filters: int,stride=2,kernel_size=3,padding=1,**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.stride=stride
        self.kernel_size=kernel_size
        self.padding=padding

    def create_layer(self,in_filters,**kwargs):
        layer=nn.Sequential(
            nn.Conv2d(in_filters, self.filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            self.act,
            NORM_DICT[self.norm](self.filters),
            nn.Dropout(p=self.dropout)
        )
        return layer


class CnnLayerT(BaseLayer):
    def __init__(self, filters: int,stride=2,kernel_size=4,padding=1,**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.stride=stride
        self.kernel_size=kernel_size
        self.padding=padding

    def create_layer(self,in_filters,**kwargs):
        layer=nn.Sequential(
            nn.ConvTranspose2d(in_filters, self.filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            self.act,
            NORM_DICT[self.norm](self.filters),
            nn.Dropout(p=self.dropout)
        )
        return layer

class ViTParams():
    def __init__(self, patch_size:int ,n_embd: int, n_head: int, n_layers: int,dropout=0.):
        self.n_embd=n_embd
        self.n_head=n_head
        self.n_layers=n_layers
        self.patch_size=patch_size
        self.dropout=dropout
    
