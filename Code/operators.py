import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import math

import collections.abc

from torch.distributions.bernoulli import Bernoulli

from einops import rearrange, reduce, repeat, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def rand_rotate(x, p):
    sampler = Bernoulli(probs=p)
    B, W, H, P1, P2, C = x.shape
    for i in range(B):
        for j in range(W):
            for k in range(H):
                if sampler.sample() == 1.0:
                    x[i][j][k] = rearrange(x[i][j][k], 'p1 p2 c -> p2 p1 c')
    return x

def center_rotate(x):
    B, W, H, P1, P2, C = x.shape
    WL = W/3
    WR = W * 2/3
    HL = H/3
    HR = H * 2/3
    for i in range(B):
        for j in range(W):
            if j < WL or j > WR:
                continue
            for k in range(H):
                if k < HL or k > HR:
                    continue
                x[i][j][k] = rearrange(x[i][j][k], 'p1 p2 c -> p2 p1 c')
    return x

def outer_rotate(x):
    B, W, H, P1, P2, C = x.shape
    WL = W/3
    WR = W * 2/3
    HL = H/3
    HR = H * 2/3
    for i in range(B):
        for j in range(W):
            for k in range(H):
                if j > WL and j < WR and k > HL and k < HR:
                    continue
                x[i][j][k] = rearrange(x[i][j][k], 'p1 p2 c -> p2 p1 c')
    return x

def rand_occlude(x, p):
    sampler = Bernoulli(probs=p)
    B, W, H, P1, P2, C = x.shape
    for i in range(B):
        for j in range(W):
            for k in range(H):
                if sampler.sample() == 1.0:
                    x[i][j][k] = torch.zeros([P1, P2, C], dtype=torch.int32)
    return x

def center_occlude(x):
    B, W, H, P1, P2, C = x.shape
    WL = W/3
    WR = W * 2/3
    HL = H/3
    HR = H * 2/3
    for i in range(B):
        for j in range(W):
            if j < WL or j > WR:
                continue
            for k in range(H):
                if k < HL or k > HR:
                    continue
                x[i][j][k] = torch.zeros([P1, P2, C], dtype=torch.int32)
    return x

def outer_occlude(x):
    B, W, H, P1, P2, C = x.shape
    WL = W/3
    WR = W * 2/3
    HL = H/3
    HR = H * 2/3
    for i in range(B):
        for j in range(W):
            for k in range(H):
                if j > WL and j < WR and k > HL and k < HR:
                    continue
                x[i][j][k] = torch.zeros([P1, P2, C], dtype=torch.int32)
    return x

def pgd_attack(model, images, labels, device, eps=0.3, alpha=2/255, iters=40):
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)
        # outputs.requires_grad = True
        # print(outputs.requires_grad)
        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        # cost.requires_grad = True
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Patch_Embed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        patch_dim = in_chans * patch_size * patch_size

        self.proj = nn.Linear(patch_dim, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        # torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = rearrange(x, 'b c (h p1) (w p2) -> b (w h) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        x = self.proj(x)
        x = self.norm(x)
        return x


class Patch_Embed_Rotate(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, p=0.5):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.p = p

    def forward(self, x):
        # B, C, H, W = x.shape
        # torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = rearrange(x, 'b c (h p1) (w p2) -> b w h p1 p2 c', p1 = self.patch_size, p2 = self.patch_size)
        x = rand_rotate(torch.clone(x), self.p)
        x = rearrange(x, 'b w h p2 p1 c -> b c (h p2) (w p1)')
        return x


class Patch_Embed_Center_Rotate(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        # B, C, H, W = x.shape
        # torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = rearrange(x, 'b c (h p1) (w p2) -> b w h p1 p2 c', p1 = self.patch_size, p2 = self.patch_size)
        x = center_rotate(torch.clone(x))
        x = rearrange(x, 'b w h p2 p1 c -> b c (h p2) (w p1)')
        return x


class Patch_Embed_Outer_Rotate(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        # B, C, H, W = x.shape
        # torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = rearrange(x, 'b c (h p1) (w p2) -> b w h p1 p2 c', p1 = self.patch_size, p2 = self.patch_size)
        x = outer_rotate(torch.clone(x))
        x = rearrange(x, 'b w h p2 p1 c -> b c (h p2) (w p1)')
        return x


class Patch_Embed_Blur(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, h1=8, h2=8):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.h1 = h1
        self.h2 = h2

    def forward(self, x):
        # B, C, H, W = x.shape
        # torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        rearanged = rearrange(x, 'b c (h p1) (w p2) -> b w h p1 p2 c', p1 = self.patch_size, p2 = self.patch_size)
        reduced = reduce(rearanged, 'b w h (p1 h1) (p2 h2) c-> b w h p1 p2 c', 'mean', h1 = self.h1, h2 = self.h2)
        repeated = repeat(reduced, 'b w h p1 p2 c -> b w h (p1 h2) (p2 w2) c', h2=self.h1, w2=self.h2)
        x = rearrange(repeated, 'b w h p1 p2 c -> b c (h p1) (w p2)')
        return x


class Patch_Embed_Shuffle(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, groups=4):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.groups = groups

    def forward(self, x):
        # B, C, H, W = x.shape
        # torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = rearrange(x, 'b c (h p1) (w p2) -> b w h p1 p2 c', p1 = self.patch_size, p2 = self.patch_size)
        x = rearrange(x, 'b (w1 w2) h p1 p2 c -> b (w2 w1) h p1 p2 c', w1=self.groups)
        x = rearrange(x, 'b w (h1 h2) p1 p2 c -> b w (h2 h1) p1 p2 c', h1=self.groups)
        x = rearrange(x, 'b w h p1 p2 c -> b c (h p1) (w p2)')
        return x

class Patch_Embed_Occlude(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, p=0.5):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.p = p

    def forward(self, x):
        # B, C, H, W = x.shape
        # torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = rearrange(x, 'b c (h p1) (w p2) -> b w h p1 p2 c', p1 = self.patch_size, p2 = self.patch_size)
        x = rand_occlude(torch.clone(x), self.p)
        x = rearrange(x, 'b w h p1 p2 c -> b c (h p1) (w p2)')
        return x

class Patch_Embed_Center_Occlude(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        # B, C, H, W = x.shape
        # torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = rearrange(x, 'b c (h p1) (w p2) -> b w h p1 p2 c', p1 = self.patch_size, p2 = self.patch_size)
        x = center_occlude(torch.clone(x))
        x = rearrange(x, 'b w h p1 p2 c -> b c (h p1) (w p2)')
        return x

class Patch_Embed_Outer_Occlude(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        # B, C, H, W = x.shape
        # torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = rearrange(x, 'b c (h p1) (w p2) -> b w h p1 p2 c', p1 = self.patch_size, p2 = self.patch_size)
        x = outer_occlude(torch.clone(x))
        x = rearrange(x, 'b w h p1 p2 c -> b c (h p1) (w p2)')
        return x