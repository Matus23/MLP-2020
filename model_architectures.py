import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from arg_extractor import get_args
args, device = get_args()




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

        
        #print("EncoderBasic ----------- "+str(x.shape)+str(out.shape))
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
        #print("----------------------------------------")
        return self.layer(x)

class DecoderBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(DecoderBasicBlock, self).__init__()
        if stride ==2:
            kern = 4
        else:
            kern = 3

        padd_out = 0
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kern, stride=stride,
                               padding=1,output_padding=padd_out, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1,output_padding=padd_out, bias=False)

        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        #If this a decoder block
        
        self.convShortcut = (not self.equalInOut) and nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kern, stride=stride,padding=1, bias=False) or None

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
        #print("DecoderBasic ----------- "+str(x.shape)+str(out.shape))
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
            layers.append(block(in_pl, out_pl, i == int(nb_layers)-1 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        #print("----------------------------------------")
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
        # print('MU: ',torch.min(mu),'EPS*STD: ',torch.min(eps*std),'LOGVAR: ',torch.min(logvar))
        return mu + eps*std

    def forward(self,x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out_mean = self.bn1_mean(self.relu_mean(self.block3_mean(out)))
        out_std = self.bn1_std(self.relu_std(self.block3_std(out)))

        out_sampled = self.reparameterize(out_mean,out_std)


        out_pr = self.relu(self.bn1(out_sampled))
        out_pr = F.avg_pool2d(out_pr, 8)
        out_pr = out_pr.view(-1, self.nChannels)
        return self.fc(out_pr), out_sampled, out_mean, out_std

class Decoder(nn.Module):
    def __init__(self, depth=28, widen_factor=2, dropRate=0.0, num_channels=1):
        super(Decoder, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((28 - 4) % 6 == 0)
        n = (28 - 4) / 6
        block = DecoderBasicBlock
        # 1st conv before any network block

        self.block3 = DecoderNetworkBlock(n, nChannels[3], nChannels[2], block, 2, dropRate)

        self.block2 = DecoderNetworkBlock(n, nChannels[2], nChannels[1], block, 2, dropRate)

        self.block1 = DecoderNetworkBlock(n, nChannels[1], nChannels[0], block, 1, dropRate, activate_before_residual=True)


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
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self,x,ae=True):
        pred,out = self.encoder(x)
        if ae:
            out = self.decoder(out)
            return pred, out
        else:
            return pred

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
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def forward(self,x, ae=True):
        pred, out, mu, logvar = self.encoder(x)
        if ae:
            out = self.decoder(out)
            return pred, out, mu, logvar
        else:
            return pred


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous()


class VQVAE_Encoder(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0, num_channels=1):
        super(VQVAE_Encoder, self).__init__()
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

        self._vq_vae = VectorQuantizer(num_embeddings=512, embedding_dim=128,
                                       commitment_cost=0.25)

        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    def forward(self,x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out_re = self.block3(out)

        loss, quantized = self._vq_vae(out_re)
        if args.vqorder == 'after':
            out_pr = self.relu(self.bn1(quantized))

        elif args.vqorder == 'before':

            out_pr = self.relu(self.bn1(out_re))
        out_pr = F.avg_pool2d(out_pr, 8)
        out_pr = out_pr.view(-1, self.nChannels)

        return self.fc(out_pr), loss, quantized


class VQVAE(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0, num_channels=1):
        super(VQVAE, self).__init__()
        self.encoder = VQVAE_Encoder(num_classes, depth, widen_factor, dropRate, num_channels)
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
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def forward(self,x, ae=True):
        pred, loss, quantized = self.encoder(x)
        if ae:
            out = self.decoder(quantized)
            return pred, out, loss
        else:
            return pred