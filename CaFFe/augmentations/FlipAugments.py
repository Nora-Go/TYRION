import torch
import torchvision
from torch import nn
import numpy as np

"""
    Augments for both Images + Masks
"""


class FlipAugments(nn.Module):
    def __init__(self, p_flip_v=0.,
                 p_rotate=0.,
                 p_flip_h=0.,
                 ):
        self.p_flip_v = p_flip_v
        self.p_rotate = p_rotate
        self.p_flip_h = p_flip_h

    def __call__(self, target_image, target_mask):
        if self.p_flip_h > torch.rand(1):
            target_image, target_mask = self.Fliph(target_image, target_mask)

        if self.p_flip_v > torch.rand(1):
            target_image, target_mask = self.Flipv(target_image, target_mask)

        if self.p_rotate > torch.rand(1):
            target_image, target_mask = self.Rotate(target_image, target_mask)

        return target_image, target_mask

    def Fliph(self, target_image, target_mask):
        target = torchvision.transforms.functional.hflip(target_image)
        mask_target = torchvision.transforms.functional.hflip(target_mask)
        return target, mask_target

    def Flipv(self, target_image, target_mask):
        target = torchvision.transforms.functional.vflip(target_image)
        mask_target = torchvision.transforms.functional.vflip(target_mask)
        return target, mask_target

    def Rotate(self, target_image, target_mask):
        random = np.random.randint(0, 3)
        angle = 90
        if random == 1:
            angle = 180
        elif random == 2:
            angle = 270
        target = torchvision.transforms.functional.rotate(target_image, angle=angle)
        mask_target = torchvision.transforms.functional.rotate(target_mask.unsqueeze(dim=0), angle=angle)[0]

        return target, mask_target.squeeze(0)
