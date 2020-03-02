import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model_architectures import VAE,AE

class AUTOENCODER(nn.Module):
    def __init__(self, depth, num_channels,widen_factor,dropRate):
        super(AUTOENCODER, self).__init__()
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        
        
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        EnBlock = EncoderBasicBlock
        DeBlock = DecoderBasicBlock
        print("Number of channels",nChannels)

        # 1st conv before any network block
        self.enconv1 = nn.Conv2d(num_channels, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.enblock1 = EncoderNetworkBlock(n, nChannels[0], nChannels[1], EnBlock, 1, dropRate, activate_before_residual=True)
        self.deblock1 = DecoderNetworkBlock(n, nChannels[1], nChannels[0], DeBlock, 1, dropRate, activate_before_residual=True)
        self.deconv1 = nn.ConvTranspose2d(nChannels[0],num_channels, kernel_size=3, stride=1,padding=1, bias=False)

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
        print("Convolution--------------------------------"+str(x.shape))
        out = self.enconv1(x)
        out = self.enblock1(out)
        out = self.deblock1(out)
        print("DeConvolution--------------------------------"+str(out.shape))
        out = self.deconv1(out)
        return out

def main():
    print("TESTING")
    depth=28
    vae = AE(10,depth,2,0.0,3)
    vae = vae.float()
   
    x = torch.randn(64,3,32,32)
    out = vae.forward(x)
  

if __name__ == "__main__":
    main()