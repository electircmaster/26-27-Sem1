import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, config, attention_fn):
        super().__init__()
        d, self.heads = config['d_model'], config['num_heads']
        self.attention_fn = attention_fn
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d,d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d,config['d_ff']),nn.GELU(),nn.Linear(config['d_ff'],d))

    def forward(self,x,allowed_mask):
        b,t,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        context,_=self.attention_fn(q,k,v,allowed_mask)
        x=x+self.proj(context.transpose(1,2).reshape(b,t,d))
        return x+self.ff(self.norm2(x))


class TinyTransformer(nn.Module):
    """Fixed architecture; callbacks support exercises and trusted format training."""
    def __init__(self, config, attention_fn, mask_fn):
        super().__init__()
        self.config=dict(config)
        self.mask_fn=mask_fn
        self.token_embedding=nn.Embedding(config['vocab_size'],config['d_model'])
        self.position_embedding=nn.Embedding(config['max_length'],config['d_model'])
        self.blocks=nn.ModuleList([Block(config,attention_fn) for _ in range(config['num_layers'])])
        self.norm=nn.LayerNorm(config['d_model'])
        self.head=nn.Linear(config['d_model'],config['vocab_size'])

    def forward(self,input_ids):
        b,t=input_ids.shape
        if t>self.config['max_length']:
            raise ValueError('Sequence exceeds max_length.')
        positions=torch.arange(t,device=input_ids.device)
        x=self.token_embedding(input_ids)+self.position_embedding(positions)[None]
        mask=self.mask_fn(t,input_ids.device)
        for block in self.blocks:x=block(x,mask)
        return self.head(self.norm(x))
