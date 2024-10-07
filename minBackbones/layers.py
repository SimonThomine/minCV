import torch.nn as nn



class BaseLayer:
    def __init__(self,act=nn.ReLU(),bn=True,dropout=False,dropout_p=0.5):
        self.act=act
        self.bn=bn
        self.dropout=dropout
        self.dropout_p=dropout_p

class MlpLayer(BaseLayer):
    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim=hidden_dim

class CnnLayer(BaseLayer):
    def __init__(self, filters: list[int],stride=2,kernel_size=3,padding=1,**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.stride=stride
        self.kernel_size=kernel_size
        self.padding=padding
        
    
