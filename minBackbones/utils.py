import torch
import torch.nn as nn
import torch.nn.functional as F
from minBackbones.layers import BaseLayer,CnnLayer,MlpLayer, CnnLayerT,ViTParams

class Head_enc(nn.Module):

    def __init__(self, head_size,n_embd,dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _,_,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size,n_embd,dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head_enc(head_size,n_embd,dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):

    def __init__(self, n_embd,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, n_embd, n_head,dropout=0.):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size,n_embd,dropout)
        self.ffwd = FeedFoward(n_embd,dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
def image_to_patches(image, patch_size):
    B,C,_,_ = image.shape
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0,2, 3, 1, 4, 5).contiguous()
    patches = patches.view(B,-1, C, patch_size, patch_size)
    patches_flat = patches.flatten(2, 4)
    return patches_flat


def verify_model_family(layers,model_family):
    if not layers:
        raise ValueError("layers must be provided to create a model")
    if not model_family:
        print("model_family not provided, trying to infer it from the layers")
    
    if isinstance(layers, ViTParams) :
        if model_family and model_family!="vit":
            raise ValueError("model_family is not consistent with the layers")
        if not model_family:
            model_family="vit"
            print("model_family inferred as vit")
    elif all(isinstance(layer, MlpLayer) for layer in layers):
        if model_family and model_family!="mlp":
            raise ValueError("model_family is not consistent with the layers")
        if not model_family:
            model_family="mlp"
            print("model_family inferred as mlp")
    elif any(isinstance(layer, CnnLayer) for layer in layers):
        if model_family and model_family!="cnn":
            raise ValueError("model_family is not consistent with the layers")
        if not model_family:
            model_family="cnn"
            print("model_family inferred as cnn")
    return model_family
    