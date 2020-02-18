import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Wide ResNet model copied from MixMatch PyTorch implementation
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False,decoder=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        #If this a decoder block
        if decoder:
            self.convShortcut = (not self.equalInOut) and nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride,
            padding=0, bias=False) or None
        else:
            self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
            padding=0, bias=False) or None

        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        # Main idea: If activate_before_residual is True then we run BatchNorm2d and LeakyReLU on the input: x
        # If equalInOut is false and activate_before_residual is true
        if not self.equalInOut and self.activate_before_residual == True:
            # Process x with BatchNorm2d and LeakyReLU
            x = self.relu1(self.bn1(x))
        else:
            # Set out to be the result of BatchNorm2d and LeakyReLU
            out = self.relu1(self.bn1(x))

        # Set out to be the result of Conv2d, BatchNorm2d, LeakyReLU of out if same size else x
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)



class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False,decoder=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual, decoder)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual,decoder):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual, decoder))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0, num_channels=1):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(num_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block for mean
        self.block3_mean = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # 3rd block for std
        self.block3_std = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)


        self.bn1_mean = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu_mean = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.bn1_std = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu_std = nn.LeakyReLU(negative_slope=0.1, inplace=True)


        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.decoder = Decoder(nChannels[3], num_channels,widen_factor,dropRate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, ae=True):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out_mean = self.block3_mean(out)
        out_std = self.block3_std(out)
        out_mean = self.relu_mean(self.bn1_mean(out_mean))
        out_std = self.relu_std(self.bn1_std(out_std))
        # print("after relu", out.size())
        out_mean = F.avg_pool2d(out_mean, 8)
        out_std = F.avg_pool2d(out_std, 8)

        out_sampled = self.reparameterize(out_mean,out_std)

        # print("after pooling", out.size())
        out = out_sampled.view(-1, self.nChannels)
        if ae:
            return self.fc(out), self.decoder(out)
        else:
            return self.fc(out)

class Decoder(nn.Module):
    def __init__(self, nChannels, num_channels,widen_factor,dropRate):
        super(Decoder, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((28 - 4) % 6 == 0)
        n = (28 - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, decoder=True)

        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, decoder=True)

        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True, decoder=True)

        self.conv1 = nn.ConvTranspose2d(num_channels, nChannels[0], kernel_size=3, stride=1,padding=1, bias=False)

    def forward(self,x):
        out = self.block3(out)
        out = self.block2(out)
        out = self.block1(out)
        out = self.conv1(x)
        return out


"""
class Decoder(nn.Module):
    def __init__(self, nChannels, num_channels):
        super(Decoder, self).__init__()
        self.dfc3 = nn.Linear(nChannels, 4096)
        self.bn3 = nn.BatchNorm1d(4096)
        self.dfc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.dfc1 = nn.Linear(4096, 256 * 4 * 4)
        self.bn1 = nn.BatchNorm1d(256 * 4 * 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding=0)
        self.dconv4 = nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.dconv3 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.dconv2 = nn.ConvTranspose2d(128, 64, 5, padding=4)
        self.dconv1 = nn.ConvTranspose2d(64, num_channels, 5, padding=2)


    def forward(self, x):
        x = self.dfc3(x)

        x = F.relu(self.bn3(x))

        x = self.dfc2(x)
        x = F.relu(self.bn2(x))

        x = self.dfc1(x)
        x = F.relu(self.bn1(x))

        # print(x.size())
        x = x.view(64, 256, 4, 4)
        # print (x.size())
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv5(x)
        # print x.size()
        x = F.relu(x)
        # print x.size()
        x = F.relu(self.dconv4(x))
        # print x.size()
        x = F.relu(self.dconv3(x))
        # print x.size()
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv2(x)
        # print x.size()
        x = F.relu(x)
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv1(x)
        # print x.size()
        # x = F.sigmoid(x)
        # print x
        return x
"""
