"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import torch.nn as nn
import torch
from torchvision import models
import sys

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input


class VideoModel(nn.Module):
    def __init__(self, num_channels=3):
        super(VideoModel, self).__init__()
        self.r2plus1d = models.video.r2plus1d_18(pretrained=True)
        self.r2plus1d.fc = nn.Sequential(nn.Dropout(0.0),
                                         nn.Linear(in_features=self.r2plus1d.fc.in_features, out_features=17))
        if num_channels == 4:
            new_first_layer = nn.Conv3d(in_channels=4,
                                        out_channels=self.r2plus1d.stem[0].out_channels,
                                        kernel_size=self.r2plus1d.stem[0].kernel_size,
                                        stride=self.r2plus1d.stem[0].stride,
                                        padding=self.r2plus1d.stem[0].padding,
                                        bias=False)
            # copy pre-trained weights for first 3 channels
            new_first_layer.weight.data[:, 0:3] = self.r2plus1d.stem[0].weight.data
            self.r2plus1d.stem[0] = new_first_layer
        self.modes = ["clip"]

    def forward(self, x):
        return self.r2plus1d(x)


class AudioModel(nn.Module):
    def __init__(self, pretrained=False):
        super(AudioModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.0),
                                       nn.Linear(in_features=self.resnet.fc.in_features, out_features=17))

        old_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, out_channels=self.resnet.conv1.out_channels,
                                      kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if pretrained == True:
            self.resnet.conv1.weight.data.copy_(torch.mean(old_layer.weight.data, dim=1, keepdim=True)) # mean channel

        self.modes = ["audio_features"]

    def forward(self, x):
        return self.resnet(x)


class TwoStreamAuralVisualModel(nn.Module):
    def __init__(self, num_channels=3, audio_pretrained=False):
        super(TwoStreamAuralVisualModel, self).__init__()
        self.audio_model = AudioModel(pretrained=audio_pretrained)
        self.video_model = VideoModel(num_channels=num_channels)
        self.fc = self.fc = nn.Sequential(nn.Dropout(0.0),
                                          nn.Linear(in_features=self.audio_model.resnet.fc._modules['1'].in_features +
                                                                self.video_model.r2plus1d.fc._modules['1'].in_features,
                                                    out_features=17))
        self.modes = ['clip', 'audio_features']
        self.audio_model.resnet.fc = Dummy()
        self.video_model.r2plus1d.fc = Dummy()

    def forward(self, audio, clip):
        #audio = x['audio_features']
        #clip = x['clip']
        audio_model_features = self.audio_model(audio)
        video_model_features = self.video_model(clip)
        print(audio_model_features.shape)
        print(video_model_features.shape)
        sys.exit()
        features = torch.cat([audio_model_features, video_model_features], dim=1)
        out = self.fc(features)
        return audio_model_features, video_model_features, out
