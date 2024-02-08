import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

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
class DASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(DASPPConv, self).__init__(*modules)


class DASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(DASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class DASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_reates):
        super(DASPP, self).__init__()
        models = []
        models.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))


        for rate_item in atrous_reates:  
            for i in range(0,3):
                models.append(DASPPConv((i+1)*in_channels,out_channels,rate_item[i]))

        models.append(DASPPPooling(in_channels, out_channels))


        self.convs = nn.ModuleList(models)

        self.down_1 = nn.Sequential(
            nn.Conv2d(4*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.down_2 = nn.Sequential(
            nn.Conv2d(5* out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):

        a_1 = self.convs[1](x)
        a_1_1=torch.cat((a_1,x),dim=1)

        a_2 = self.convs[4](x)
        a_2_2=torch.cat((a_2,x),dim=1)

        a_3 = self.convs[7](x)
        a_3_3=torch.cat((a_3,x),dim=1)

        for i in range(1, 3):
            a_1=self.convs[1+i](a_1_1)
            a_1_1=torch.cat((a_1,a_1_1),dim=1)

            a_2=self.convs[4+i](a_2_2)
            a_2_2=torch.cat((a_2,a_2_2),dim=1)

            a_3=self.convs[7+i](a_3_3)
            a_3_3=torch.cat((a_3,a_3_3),dim=1)

        a_4 = self.convs[0](x)
        a_5 = self.convs[10](x)


        res = torch.cat((self.down_1(a_1_1),self.down_1(a_2_2),self.down_1(a_3_3), a_4, a_5), dim=1)  
        out = self.relu(x + self.down_2(res))  
        return out


class UNET_DASPP(nn.Module):
    def __init__(
            self, in_channels, out_channels, features=[64, 128, 256, 512],
    ):
        super(UNET_DASPP, self).__init__()
        self.ups = nn.ModuleList()  
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu=nn.ReLU(inplace=True)


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

        rate = [[4, 6, 12], [6, 12, 18], [12, 18, 24]]

        
        self.DAspp=DASPP(1024,1024,rate)

    def forward(self, x):
        skip_connections = []

        # decoder part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x=self.relu(x+self.DAspp(x))

        skip_connections = skip_connections[::-1]

        # encoder part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


