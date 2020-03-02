import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class EncoderBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(EncoderBasicBlock, self).__init__()
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

        print("EncoderBasic ----------- "+str(x.shape)+str(out.shape))
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class EncoderNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(EncoderNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        
        return self.layer(x)

class DecoderBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(DecoderBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        #If this a decoder block
        
        self.convShortcut = (not self.equalInOut) and nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride,
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
        
        print("DecoderBasic ----------- "+str(x.shape)+str(out.shape))
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class DecoderNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(DecoderNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            in_pl = in_planes
            out_pl = (i == int(nb_layers)-1 and out_planes or in_planes)
            layers.append(block(in_pl, out_pl, i == int(nb_layers) and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

# Wide ResNet model copied from MixMatch PyTorch implementation
class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0, num_channels=1):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = EncoderBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(num_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = EncoderNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = EncoderNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block for mean
        self.block3 = EncoderNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        self.fc = nn.Linear(nChannels[3], num_classes)

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
        out = self.block3(out)
        return self.fc(out)       

class AE_Encoder(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0, num_channels=1):
        super(AE_Encoder, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = EncoderBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(num_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = EncoderNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = EncoderNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block for mean
        self.block3 = EncoderNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    def forward(self,x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out_re = self.block3(out)

        out_pr = self.relu(self.bn1(out_re))
        out_pr = F.avg_pool2d(out_pr, 8)
        out_pr = out_pr.view(-1, self.nChannels)

        return self.fc(out_pr), out_re

class VAE_Encoder(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0, num_channels=1):
        super(VAE_Encoder, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = EncoderBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(num_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = EncoderNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = EncoderNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block for mean
        self.block3_mean = EncoderNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # 3rd block for std
        self.block3_std = EncoderNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        self.bn1_mean = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu_mean = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.bn1_std = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu_std = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self,x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out_mean = self.block3_mean(out)
        out_std = self.block3_std(out)

        out_sampled = self.reparameterize(out_mean,out_std)

        out_pr = self.relu(self.bn1(out_sampled))
        out_pr = F.avg_pool2d(out_pr, 8)
        out_pr = out_pr.view(-1, self.nChannels)
        return self.fc(out_pr), out_sampled

class Decoder(nn.Module):
    def __init__(self, depth=28, widen_factor=2, dropRate=0.0, num_channels=1):
        super(Decoder, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((28 - 4) % 6 == 0)
        n = (28 - 4) / 6
        block = DecoderBasicBlock
        # 1st conv before any network block
        self.block3 = DecoderNetworkBlock(n, nChannels[3], nChannels[2], block, 2, dropRate, activate_before_residual=True)

        self.block2 = DecoderNetworkBlock(n, nChannels[2], nChannels[1], block, 2, dropRate)

        self.block1 = DecoderNetworkBlock(n, nChannels[1], nChannels[0], block, 1, dropRate)

        self.conv1 = nn.ConvTranspose2d(nChannels[0], num_channels, kernel_size=3, stride=1,padding=1, bias=False)

    def forward(self,x):
        out = self.block3(x)
        out = self.block2(out)
        out = self.block1(out)
        out = self.conv1(out)
        return out

class AE(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0, num_channels=1):
        super(AE, self).__init__()
        self.encoder = AE_Encoder(num_classes, depth, widen_factor, dropRate, num_channels)
        self.decoder = Decoder(depth, widen_factor, dropRate, num_channels)

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

    def forward(self,x):
        pred,out = self.encoder(x)
        out = self.decoder(out)
        return pred,out

class VAE(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0, num_channels=1):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(num_classes, depth, widen_factor, dropRate, num_channels)
        self.decoder = Decoder(depth, widen_factor, dropRate, num_channels)

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

    def forward(self,x):
        pred,out = self.encoder(x)
        out = self.decoder(out)
        return pred,out