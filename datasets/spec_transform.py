"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import torch
from torchvision.transforms.functional import normalize
import numpy as np
import cv2
import sys
import random
import importlib
from torchaudio.transforms import AmplitudeToDB

class ComposeWithInvert(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, invert=False):
        if invert:
            for t in reversed(self.transforms):
                img = t(img, invert)
        else:
            for t in self.transforms:
                img = t(img, invert)
        return img


class NumpyToTensor:
    #convert numpy to tensor, or tensor to tensor
    def __init__(self):
        pass

    def __call__(self, clip, invert):

        if invert:
            print("inverting")
            #convert to TODO
            clip = clip.permute(1, 2, 3, 0)
            clip = clip.mul(255).to(torch.uint8)
        else:
            #convert from img int8 T, W, H, C to float 0-1 image C T W H
            clip = clip.astype(np.float32) / 255
            clip = torch.from_numpy(clip).permute(3, 0, 1, 2)

        return clip


class Normalize:
    """Normalize an tensor image or video clip with mean and standard deviation.
       Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.

        # forward is an in place operation!
        # invert is not
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.mean_t = None
        self.std_t = None

    def __call__(self, clip, invert):
        if self.mean_t is None:
            dtype = clip.dtype
            if len(clip.shape) == 4:
                self.mean_t = torch.as_tensor(self.mean, dtype=dtype, device=clip.device)[:, None, None, None]
                self.std_t = torch.as_tensor(self.std, dtype=dtype, device=clip.device)[:, None, None, None]
            else:
                self.mean_t = torch.as_tensor(self.mean, dtype=dtype, device=clip.device)[:, None, None]
                self.std_t = torch.as_tensor(self.std, dtype=dtype, device=clip.device)[:, None, None]

        if invert:
            clip = clip.clone()
            clip.mul_(self.std_t).add_(self.mean_t)
        else:
            # image of size (C, H, W) to be normalized.
            #clip = normalize(clip, self.mean, self.std)
            clip.sub_(self.mean_t).div_(self.std_t)

        return clip


class AmpToDB:

    def __init__(self):
        self.amplitude_to_DB = AmplitudeToDB('power', 80)

    def __call__(self, features, invert):

        if invert:
            pass # do nothing
        else:
            features = self.amplitude_to_DB(features)

        return features


class RandomClipFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip, invert):

        if invert:
            pass # do nothing
        else:
            if random.random() < self.p:
                # T W H C
                #assert clip.shape[3] == 3 # last channel is RGB
                # for every image apply cv2 flip
                for i in range(clip.shape[0]):
                    clip[i] = cv2.flip(clip[i], 1)

        return clip
