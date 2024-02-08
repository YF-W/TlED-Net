import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import warnings

warnings.filterwarnings("ignore")

class CSMAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CSMAttention, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(2*in_channels, 2*in_channels // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(2*in_channels // 16, in_channels, 1, bias=False))
        self.fc_2 = nn.Sequential(nn.Conv2d(2*in_channels, 2*in_channels // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(2*in_channels // 16, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def _process_1(self,x,eps=1e-5):      
        N, C, _, _ = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)[:,:,None]	
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps	
        channel_std = channel_var.sqrt()[:,:,None]		

        t = torch.cat((channel_mean, channel_std), dim=1)
        t=self.fc(t)
        t=self.bn(t)
        return self.sigmoid(t)

    def forward(self, x):
        a_max=self.max_pool(x)
        a_all=torch.cat((a_max,self._process_1(x)),dim=1)
        a_all=self.sigmoid(self.fc_2(a_all))
        out=x*a_all

        return out



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class UNET_CSMA(nn.Module):
    def __init__(
            self, in_channels, out_channels, features=[64, 128, 256, 512],
    ):
        super(UNET_CSMA, self).__init__()
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)




        self.CSs=nn.ModuleList()
        self.CSs.append(CSMAttention(1024))
        self.CSs.append(CSMAttention(512))
        self.CSs.append(CSMAttention(256))
        self.CSs.append(CSMAttention(128))



    def forward(self, x):
        skip_connections = []

        # encoder part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # decoder part
        flag=0
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](self.CSs[flag](concat_skip))
            flag+=1

        return self.final_conv(x)

