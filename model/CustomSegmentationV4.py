import torch
import torch.nn as nn


# Segmentation version 4

class residual_block(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, activation=torch.nn.SiLU):
        super(residual_block, self).__init__()

        self.stride = stride
        self.features = torch.nn.Sequential(torch.nn.Conv2d(in_channels,
                                                            out_channels,
                                                            kernel_size=3,
                                                            stride=self.stride,
                                                            padding=1,
                                                            bias=False),
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
        self.layer = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               dilation=1,
                               bias=False,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
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
                                        dilation=1,
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


class CustomSegmentationV4(torch.nn.Module):

    def __init__(self):
        super(CustomSegmentationV4, self).__init__()


        self.exapand_rate = 3

        #200x56x32
        self.down1 = Down(in_channel=3, out_channel=8, stride=2, expand_ratio=1) #32

        # 100x28x64
        self.down2 = Down(in_channel=8, out_channel=16, stride=2, expand_ratio=self.exapand_rate) #64

        # 50x14x128
        self.down3 = Down(in_channel=16, out_channel=32, stride=2, expand_ratio=self.exapand_rate) #128

        # 25x7x256
        self.down4 = Down(in_channel=32, out_channel=64, stride=2, expand_ratio=self.exapand_rate) #256


        # 25x7x256
        self.center1 = residual_block(in_channels=64, out_channels=64) #256
        self.center2 = residual_block(in_channels=64, out_channels=64) #256



        # 50x14x128
        self.up3 = Up(in_channel=64, out_channel=32, stride=2, expand_ratio=self.exapand_rate)  #128


        # 100x28x64
        self.up2 = Up(in_channel=32, out_channel=16, stride=2, expand_ratio=self.exapand_rate)   #64

        #200x56x32
        self.up1 = Up(in_channel=16, out_channel=8, stride=2, expand_ratio=self.exapand_rate)    #32

        self.final1 = nn.ConvTranspose2d(in_channels=8,
                                         out_channels=8,
                                         dilation=1,
                                         bias=True,
                                         kernel_size=2,
                                         stride=2)
        self.final1_relu6 = nn.SiLU()
        self.final2 = nn.Conv2d(in_channels=8,
                                out_channels=1,
                                bias=True,
                                kernel_size=3,
                                padding='same')

        #self.final2_bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        


    def forward(self, x):
        down1 = self.down1(x)
        
        down2 = self.down2(down1)
        
        down3 = self.down3(down2)
        
        down4 = self.down4(down3)

        center = self.center1(down4)
        center = self.center2(center)

        up3 = self.up3(center)
        sum3 = torch.add(up3, down3)

        up2 = self.up2(sum3) #
        sum2 = torch.add(up2, down2)

        up1 = self.up1(sum2)
        sum1 = torch.add(up1, down1)

        final1 = self.final1(sum1)
        final1_relu6 = self.final1_relu6(final1)
        final2 = self.final2(final1_relu6)
        y = self.sigmoid(final2)
        
        return y


