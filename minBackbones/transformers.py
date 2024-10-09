import torch
import torch.nn as nn
from minBackbones.utils import TransformerBlock,image_to_patches


class ViT(nn.Module):
    def __init__(self,image_dim, layers,type="classification",**kwargs) -> None:
        super().__init__()
        C,H,W=image_dim
        self.patch_size=layers.patch_size
        n_embed=layers.n_embd
        n_head=layers.n_head
        n_layer=layers.n_layers
        dropout=layers.dropout
        assert H%self.patch_size==0 and W%self.patch_size==0, "Image Dim must be divisible by patch size"
        nb_patches=(image_dim[1]//self.patch_size)*(image_dim[2]//self.patch_size)

        self.proj_layer = nn.Linear(C*self.patch_size*self.patch_size, n_embed)
        self.pos_emb = nn.Embedding(nb_patches+1, n_embed)
        self.register_parameter(name='cls_token', param=torch.nn.Parameter(torch.zeros(1, 1, n_embed)))
        self.transformer=nn.Sequential(*[TransformerBlock(n_embed, n_head,dropout) for _ in range(n_layer)])
        if type=="classification":
            classes=kwargs.get("classes")
            self.classi_head = nn.Linear(n_embed, classes if classes>2 else 1)
    
    def forward(self,x):
        B,_,_,_=x.shape
        x = image_to_patches(x, self.patch_size)
        x = self.proj_layer(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_emb = self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = x + pos_emb
        x = self.transformer(x)
        cls_tokens = x[:, 0]
        x = self.classi_head(cls_tokens)
        return x.squeeze()