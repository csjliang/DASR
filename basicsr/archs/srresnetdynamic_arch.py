import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBNDynamic, make_layer, Dynamic_conv2d

@ARCH_REGISTRY.register()
class MSRResNetDynamic(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_models=5, upscale=4):
        super(MSRResNetDynamic, self).__init__()
        self.upscale = upscale

        self.conv_first = Dynamic_conv2d(num_in_ch, num_feat, 3, groups=1, if_bias=True, K=num_models)
        self.body = make_layer(ResidualBlockNoBNDynamic, num_block, num_feat=num_feat, num_models=num_models)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = Dynamic_conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, groups=1, if_bias=True, K=num_models)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = Dynamic_conv2d(num_feat, num_feat * 4, 3, groups=1, if_bias=True, K=num_models)
            self.upconv2 = Dynamic_conv2d(num_feat, num_feat * 4, 3, groups=1, if_bias=True, K=num_models)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = Dynamic_conv2d(num_feat, num_feat, 3, groups=1, if_bias=True, K=num_models)
        self.conv_last = Dynamic_conv2d(num_feat, num_out_ch, 3, groups=1, if_bias=True, K=num_models)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x, weights):
        out = self.lrelu(self.conv_first({'x': x, 'weights': weights}))
        out = self.body({'x': out, 'weights': weights})['x']

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1({'x': out, 'weights': weights})))
            out = self.lrelu(self.pixel_shuffle(self.upconv2({'x': out, 'weights': weights})))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1({'x': out, 'weights': weights})))

        out = self.lrelu(self.conv_hr({'x': out, 'weights': weights}))
        out = self.conv_last({'x': out, 'weights': weights})
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out