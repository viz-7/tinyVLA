import math
import torch
import torch.nn as nn
from torch.nn.modules import ReLU
from tokenizer import FAST
import torch.nn.functional as F

class Config:
  D = 128
  H = 8
  T = 100
  BASE = 1000
  TEMP = 0.7
  def __init__(self) -> None:
    assert(self.D % self.H == 0)
    assert((self.D // self.H) % 2 == 0)

class RoPe(nn.Module):
  def __init__(self, config: Config) -> None:
    super().__init__()
    self.config = config
    d = self.config.D // self.config.H
    t = self.config.T
    theta = 1 /(self.config.BASE ** (torch.arange(0, d, 2).float() / d))
    seq = torch.arange(0, t)
    theta_2 = torch.einsum("n,d->nd", seq, theta)
    theta_2 = torch.cat([theta_2, theta_2], dim=-1).view(1, 1, t, d)
    self.register_buffer("sin", theta_2.sin())
    self.register_buffer("cos", theta_2.cos())
  def forward(self, x):
    B, H, T, D = x.shape
    d_2 = D // 2
    x_neg = torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)
    return (x * self.cos[:, :, :T]) + (x_neg * self.sin[:, :, :T])

class causal_self_attention(nn.Module):
  def __init__(self, rope: RoPe, config: Config):
    super().__init__()
    self.config = config
    self.rope = rope
    self.QKV_proj = nn.Linear(config.D, config.D*3)
    self.out_proj = nn.Linear(config.D, config.D)
    T = config.T
    self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))
  def forward(self, x):
    B, T, D = x.shape
    q,k,v = self.QKV_proj(x).split(self.config.D, dim=2)
    h = self.config.H
    d = self.config.D // h
    k = k.view(B, T, h, d).transpose(1, 2) # [B, H, T, d // H]
    q = q.view(B, T, h, d).transpose(1, 2)
    v = v.view(B, T, h, d).transpose(1, 2)
    q = self.rope(q)
    k = self.rope(k)

    attn = (q @ k.transpose(-2, -1))/math.sqrt(d) # [B, H, T, T]
    attn = attn.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    y = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)

    return self.out_proj(y)

class block(nn.Module):
  def __init__(self, rope: RoPe, config: Config) -> None:
    super().__init__()
    d = config.D
    self.norm1 = nn.LayerNorm(d)
    self.norm2 = nn.LayerNorm(d)
    self.attn  = causal_self_attention(rope, config)
    self.ffn = nn.Sequential(
        nn.Linear(d, d*4),
        nn.ReLU(),
        nn.Linear(d*4, d*4),
        nn.ReLU(),
        nn.Linear(d*4, d)
    )
  def forward(self, x):
    x = x + self.attn(self.norm1(x))
    return x + self.ffn(self.norm2(x))

class RGBEncoder(nn.Module):
  def __init__(self, config: Config) -> None:
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, config.D)
    )
  def forward(self, x):
    x = x.permute(0, 3, 1, 2).float() / 255.0
    return self.encoder(x)

class VLA(nn.Module):
  def __init__(self, tok: FAST, config: Config) -> None:
    super().__init__()
    rope = RoPe(config)
    self.config = config
    self.tok = tok
    self.emb_size = tok.vocab_size
    self.emb = nn.Embedding(self.emb_size, config.D)
    self.rgb_enc = RGBEncoder(config)
    self.blocks = nn.Sequential(
        block(rope, config),
        block(rope, config),
        block(rope, config)
      )
    self.head = nn.Sequential(
        nn.LayerNorm(config.D),
        nn.Linear(config.D, config.D*3),
        nn.ReLU(),
        nn.Linear(config.D*3, config.D*3),
        nn.ReLU(),
        nn.Linear(config.D*3, self.emb_size),
      )
  def encoder_img(self, img): return self.rgb_enc(img).unsqueeze(1)
  def forward(self, x, img_enc):
    x = self.emb(x)
    x = self.blocks(torch.cat([img_enc, x], dim=-2))
    return self.head(x)
  def sample(self, logits):
    with torch.no_grad():
      logits = logits[:, -1, :] / self.config.TEMP
      probs = F.softmax(logits, dim=-1)
      return torch.multinomial(probs, num_samples=1)
