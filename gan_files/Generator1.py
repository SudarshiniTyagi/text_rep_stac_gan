import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchnet as tnt

import numpy as np

import cv2


class Generator1(nn.Module):
    def __init__(self):
        super(Generator1, self).__init__()
        self.relu = nn.LeakyReLU(True)
        self.c_dim = 128
        self.ngf = 192*8

        self.fc = nn.Sequential(
            nn.Linear(228, 192 * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(192 * 8 * 4 * 4),
            nn.LeakyReLU(True))

        self.ca_fc = nn.Linear(512, 256, bias= True)
        self.ca_relu = nn.LeakyReLU()

        self.upscale1 = self.upscale(self.ngf, self.ngf // 2)
        self.upscale2 = self.upscale(self.ngf//2, self.ngf//4)
        self.upscale3 = self.upscale(self.ngf//4, self.ngf//8)
        self.upscale4 = self.upscale(self.ngf//8, self.ngf//16)
        self.gen_img = nn.Sequential(
            self.conv3x3(self.ngf // 16, 3),
            nn.Tanh())

    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def upscale(self, in_planes, out_planes):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            self.conv3x3(in_planes, out_planes, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(True)
        )
        return block


    def cond_aug_network(self, text_embedding):
        
        x = self.ca_relu(self.ca_fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        normalised_x = eps.mul(std).add_(mu)

        return normalised_x, mu, logvar


    def forward(self, text_embedding):

        noise = Variable(torch.FloatTensor(text_embedding.shape[0], 100))
        noise.data.normal_(0, 1)
        text_encoding, mu, logvar = self.cond_aug_network(text_embedding)
        text_noise_encoding = torch.cat((noise, text_encoding), 1)

        h_code = self.fc(text_noise_encoding)

        h_code = h_code.view(-1, self.ngf, 4, 4)
        h_code = self.upscale1(h_code)
        h_code = self.upscale2(h_code)
        h_code = self.upscale3(h_code)
        h_code = self.upscale4(h_code)
        # state size 3 x 64 x 64
        generated_img = self.gen_img(h_code)
        # cv2.imwrite('gen_img.png', ((generated_img[0].data.detach().numpy().reshape(64, 64,
        #                                                                             3) * 115.2628580729166) + 84.12893371287164))
        #
        # cv2.imwrite('gen_img2.png', ((generated_img[1].data.detach().numpy().reshape(64, 64,
        #                                                                             3) * 143.04833984375) + 61.44166128438021))

        # cv2.imwrite('gen_img.png', (generated_img[0].data.detach().numpy().reshape(64, 64,3) * 255).astype(np.uint8))
        # cv2.imwrite('gen_img2.png', (generated_img[1].data.detach().numpy().reshape(64, 64,3) * 255).astype(np.uint8))

        return None, generated_img, mu, logvar