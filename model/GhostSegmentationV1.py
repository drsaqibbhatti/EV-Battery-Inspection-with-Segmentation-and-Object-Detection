import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class WSConv2d(torch.nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-4,
                 padding_mode='zeros',
                 gain=True,
                 gamma=1.0,
                 use_layernorm=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        torch.nn.init.kaiming_normal_(self.weight)
        self.gain = torch.nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        # gamma * 1 / sqrt(fan-in)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2 if use_layernorm else eps
        # experimental, slightly faster/less GPU memory use
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * \
                F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            mean = torch.mean(
                self.weight, dim=[1, 2, 3], keepdim=True)
            std = torch.std(
                self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, input):
        return F.conv2d(input, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)




class NFGhostModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, dw_size=3, stride=1, use_activation=True, activation=torch.nn.ReLU):
        super(NFGhostModule, self).__init__()
        self.oup = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            torch.nn.BatchNorm2d(init_channels),
            activation(inplace=True) if use_activation else torch.nn.Sequential(),
        )

        self.cheap_operation = torch.nn.Sequential(
            torch.nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            torch.nn.BatchNorm2d(new_channels),
            activation(inplace=True) if use_activation else torch.nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]



class Down(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=32):
        super(Down, self).__init__()
        
        self.feature = torch.nn.Sequential(
            NFGhostModule(in_channels=in_channel,
                          out_channels=out_channel,
                          stride=1),
            NFGhostModule(in_channels=out_channel,
                          out_channels=out_channel,
                          stride=1),
        )


        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, 
                                           stride=2)
            

    def forward(self, x):
        x = self.feature(x)
        down = self.max_pool(x)
    
        return x, down

    
class GhostSegmentationV1(torch.nn.Module):

    def __init__(self, num_classes=3, features=64, scal_factor=1.0):
        super(GhostSegmentationV1, self).__init__()

        self.features = int(features * scal_factor)

        # Downsampling layers
        self.down1 = Down(in_channel=3, out_channel=self.features)
        self.down2 = Down(in_channel=self.features, out_channel=self.features)
        self.down3 = Down(in_channel=self.features, out_channel=self.features)
        self.down4 = Down(in_channel=self.features, out_channel=self.features)

        self.center = torch.nn.Sequential(
            NFGhostModule(in_channels=self.features,
                          out_channels=self.features,
                          stride=1),
            NFGhostModule(in_channels=self.features,
                          out_channels=self.features,
                          stride=1),
        )

        self.final = torch.nn.Sequential(
            NFGhostModule(in_channels=self.features,
                          out_channels=self.features,
                          stride=1),
            NFGhostModule(in_channels=self.features,
                          out_channels=self.features,
                          stride=1),
            torch.nn.Conv2d(in_channels=self.features,
                            out_channels=num_classes,
                            kernel_size=1,
                            bias=True)
        )
       
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        
        feature1, down1 = self.down1(x) #640->x2
        feature2, down2 = self.down2(down1) #320->x4
        feature3, down3 = self.down3(down2) #160->x8
        feature4, down4 = self.down4(down3) #80->x16

        center = self.center(down4)

        up1 = F.interpolate(center, scale_factor=16, mode='bilinear')
        up2 = F.interpolate(feature4, scale_factor=8, mode='bilinear')
        up3 = F.interpolate(feature3, scale_factor=4, mode='bilinear')
        up4 = F.interpolate(feature2, scale_factor=2, mode='bilinear')

        fusion = up1 + up2 + up3 + up4 + feature1

        final = self.final(fusion)
        sigmod = self.sigmoid(final)
        
        return sigmod  # Return raw logits without applying Softmax
