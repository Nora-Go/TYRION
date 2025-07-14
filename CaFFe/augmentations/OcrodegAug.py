import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import adjust_brightness, adjust_gamma


def poisson_noise_more_random(input, rate=1, peak=3.0):
    final_noise_input = np.zeros(input.shape).astype(np.float32)
    for i in range(rate):
        extra_scalar_input = np.random.rand(*input.shape)
        noise_input = np.random.poisson(peak, input.shape).astype(np.float32)
        final_noise_input = final_noise_input + extra_scalar_input * (noise_input - peak)
    final_noise_input = final_noise_input / final_noise_input.max()
    input = np.clip(input + final_noise_input, a_min=0.0, a_max=1.0)
    return input


# data augmentation based on https://github.com/NVlabs/ocrodeg
class OcrodegAug(nn.Module):
    def __init__(self,
                 p_gamma=0.,
                 p_brightness=0.,
                 p_poisson_speckel=0.,
                 color_channels=1):
        super(OcrodegAug, self).__init__()

        self.toTensor = transforms.ToTensor()
        self.color_channels = color_channels
        self.p_brightness = p_brightness
        self.p_gamma = p_gamma
        self.p_poisson_speckel = p_poisson_speckel

    def __call__(self, x, skip_noise=False):
        x = np.array(x)[0]
        x = x / 255.0

        if self.p_poisson_speckel > torch.rand(1) and not skip_noise:
            repeats = 1
            peak = int(np.random.rand() * 3) + 1
            x = poisson_noise_more_random(x, repeats, peak)

        x = Image.fromarray((x * 255).astype(np.uint8))

        if self.p_brightness > torch.rand(1):
            factor = np.random.uniform(0.6, 1.4)
            x = adjust_brightness(x, factor)

        if self.p_gamma > torch.rand(1):
            factor = np.random.uniform(0.7, 1.3)
            x = adjust_gamma(x, factor)

        return np.expand_dims(np.array(x), axis=0)
