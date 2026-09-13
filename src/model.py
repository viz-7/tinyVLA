import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
  D = 128
  H = 8
  T = 100
  BASE = 1000
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
  def __init__(self, config: Config, rope: RoPe):
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
  def __init__(self) -> None:
    super().__init__()

class VLA(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.blocks = nn.Sequential(
        
      )
