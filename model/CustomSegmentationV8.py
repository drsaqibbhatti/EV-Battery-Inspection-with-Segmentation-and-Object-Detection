import torch
import torch.nn as nn


# Segmentation version 4


class separableConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, bias=True, activation=torch.nn.SiLU):
        super(separableConv, self).__init__()
        
        self.sptialwise_conv = torch.nn.Conv2d(in_channels=in_channels,
                                               out_channels=in_channels,
                                               kernel_size=kernel_size,
                                               stride=1,
                                               padding=padding,
                                               groups=in_channels,
                                               bias=bias)
        self.act1 = activation()
        self.pointwise_conv = torch.nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=stride,
                                              bias=bias)
        self.act2 = activation()
    def forward(self, x):
        x = self.sptialwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class residual_block(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, activation=torch.nn.SiLU):
        super(residual_block, self).__init__()

        self.stride = stride
        self.features = torch.nn.Sequential(torch.nn.Conv2d(in_channels,
                                                            out_channels,
                                                            kernel_size=5,
                                                            stride=self.stride,
                                                            bias=False,
                                                            padding=2),
                                            torch.nn.BatchNorm2d(num_features=out_channels))

        self.down_skip_connection = torch.nn.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=1,
                                                    stride=self.stride)
        self.dim_equalizer = torch.nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1)
        self.final_activation = activation()



    def forward(self, x):
        if self.stride == 2:
            down = self.down_skip_connection(x)
            out = self.features(x)
            out = out + down
        else:
            out = self.features(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        out = self.final_activation(out)
        return out


class Down(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=32, stride=1, expand_ratio=1):
        super(Down, self).__init__()
        self.layer = separableConv(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=5,
                                    stride=stride,
                                    padding=2)
        self.bn = nn.BatchNorm2d(out_channel)
        self.bottleneck = residual_block(in_channels=out_channel,
                                         out_channels=out_channel)
        self.relu = nn.SiLU()

    def forward(self, x):
        x = self.layer(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.bottleneck(x)
        x = self.relu(x)
        return x

class Up(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=32, stride=2, expand_ratio=1):
        super(Up, self).__init__()
        self.layer = nn.ConvTranspose2d(in_channels=in_channel,
                                        out_channels=out_channel,
                                        bias=False,
                                        kernel_size=2,
                                        stride=stride)
        self.bn = nn.BatchNorm2d(out_channel)
        self.bottleneck = residual_block(in_channels=out_channel,
                                         out_channels=out_channel)
        self.relu = nn.SiLU()

    def forward(self, x):
        x = self.layer(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.bottleneck(x)
        x = self.relu(x)

        return x


class CustomSegmentationV8(torch.nn.Module):

    def __init__(self):
        super(CustomSegmentationV8, self).__init__()


        self.exapand_rate = 3
        
        self.stem = torch.nn.Sequential(
            separableConv(in_channels=3,
                          out_channels=16,
                          kernel_size=5,
                          stride=2,
                          padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.SiLU()
        )
        #640X640x16


        self.down1 = Down(in_channel=16, out_channel=32, stride=2, expand_ratio=1)  #320X320x32

 
        self.down2 = Down(in_channel=32, out_channel=32, stride=2, expand_ratio=self.exapand_rate) #160x160x32


        self.down3 = Down(in_channel=32, out_channel=64, stride=2, expand_ratio=self.exapand_rate) #80x80x64


        self.down4 = Down(in_channel=64, out_channel=128, stride=2, expand_ratio=self.exapand_rate) #40x40x128



        self.center1 = residual_block(in_channels=128, out_channels=128) #40x40x128
        self.center2 = residual_block(in_channels=128, out_channels=128) #40x40x128



    
        self.up4 = Up(in_channel=128, out_channel=64, stride=2, expand_ratio=self.exapand_rate)  #80x80x64

        self.up3 = Up(in_channel=64, out_channel=32, stride=2, expand_ratio=self.exapand_rate)   #160x160x32

        self.up2 = Up(in_channel=32, out_channel=32, stride=2, expand_ratio=self.exapand_rate)   #320x320x32

        self.up1 = Up(in_channel=32, out_channel=16, stride=2, expand_ratio=self.exapand_rate)   #640x640x16

        self.final1 = separableConv(in_channels=16,
                                    out_channels=16,
                                    kernel_size=1,
                                    padding='same') #640x640x16
        
        self.final1_relu = torch.nn.ReLU()
        
        
        self.resize = nn.ConvTranspose2d(in_channels=16,
                                         out_channels=16,
                                         bias=False,
                                         kernel_size=2,
                                         stride=2)
        
        
        self.final2 = separableConv(in_channels=16, #640x640x1
                                    out_channels=1,
                                    kernel_size=3,
                                    padding='same')
        self.sigmoid = nn.Sigmoid()
        


    def forward(self, x):
        
        stem = self.stem(x)         #640X640x16
        #print('stem=', stem.size())

        down1 = self.down1(stem)#320X320x32
        #print('down1=', down1.size())
        
        down2 = self.down2(down1) #160x160x32
        #print('down2=', down2.size())
        
        down3 = self.down3(down2)#80x80x64
        #print('down3=', down3.size())
        
        down4 = self.down4(down3) #40x40x128
        #print('down4=', down4.size())
        
        center = self.center1(down4)#40x40x128
        center = self.center2(center)#40x40x128

        up3 = self.up4(center) #80x80x64
        sum3 = up3 + down3

        up2 = self.up3(sum3) #
        sum2 = up2 + down2

        up1 = self.up2(sum2)
        sum1 = up1 + down1
        
        up0 = self.up1(sum1)
        sum0 = up0 + stem

        final1 = self.final1(sum0)
        final1_relu6 = self.final1_relu(final1)
        
        resize = self.resize(final1_relu6)
        
        final2 = self.final2(resize)
        y = self.sigmoid(final2)
        
        return y


