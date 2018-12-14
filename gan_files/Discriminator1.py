import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchnet as tnt

import numpy as np

class Discriminator1(nn.Module):
    def __init__(self, ndf, nef):
        super(Discriminator1, self).__init__()

        self.encode_img = nn.Sequential(
            # 3 x 64 x 64-> 96 x 32 x 32
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False), #WHY??????
            nn.LeakyReLU(0.2, inplace=True),
            # 96 x 32 x 32  -> 96*2 x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 96*2 x 16 x 16 -> 96*4 x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 96*4 x 8 x 8 -> 96*8 x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.cond_logits = nn.Sequential(
                self.conv3x3(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid()
        )

    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, image, c_code = None):
        img_embedding = self.encode_img(image)

        if c_code is not None:
            c_code = c_code.view(-1, 128, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((img_embedding, c_code), 1)
        else:
            h_c_code = img_embedding

        output = self.cond_logits(h_c_code)
        return output.view(-1)