import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

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

class UNET_UUU(nn.Module):
    def __init__(
            self, in_channels, out_channels, features=[64, 128, 256, 512],
    ):
        super(UNET_UUU, self).__init__()
        self.ups = nn.ModuleList()  
        self.ups_2 = nn.ModuleList()
        self.ups_3 = nn.ModuleList()

        self.downs_1 = nn.ModuleList()
        self.downs_2 = nn.ModuleList()
        self.downs_3 = nn.ModuleList()


        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs_1.append(DoubleConv(in_channels, feature))
            in_channels = feature

        in_channels=64
        # Down part2 of UNET
        for feature in features[1:4]:
            self.downs_2.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Down part3 of UNET
        in_channels=64
        for feature in features[1:4]:
            self.downs_3.append(DoubleConv(in_channels, feature))
            in_channels = feature


        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))


        # Up part2 of UNET
        for feature in reversed(features):
            self.ups_2.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups_2.append(DoubleConv(feature * 2, feature))

        # Up part3 of UNET
        for feature in reversed(features):
            self.ups_3.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups_3.append(DoubleConv(feature * 2, feature))


        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.bottleneck_2 = DoubleConv(features[-1], features[-1] * 2)
        self.bottleneck_3 = DoubleConv(features[-1], features[-1] * 2)


        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        skip_connections_2 = []

        skip_connections_3 = []



        # encoder1 part
        for down in self.downs_1:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]





        # decoder part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        skip_connections_2.append(x)

        # encoder2 part
        for down in self.downs_2:
            x = down(x)
            skip_connections_2.append(x)
            x = self.pool(x)

        x = self.bottleneck_2(x)
        skip_connections_2 = skip_connections_2[::-1]


        # decoder2 part
        for idx in range(0, len(self.ups_2), 2):
            x = self.ups_2[idx](x)
            skip_connection2 = skip_connections_2[idx // 2]

            if x.shape != skip_connection2.shape:
                x = TF.resize(x, size=skip_connection2.shape[2:])

            concat_skip = torch.cat((skip_connection2, x), dim=1)
            x = self.ups_2[idx + 1](concat_skip)

        skip_connections_3.append(x)

        # encoder3 part
        for down in self.downs_3:
            x = down(x)
            skip_connections_3.append(x)
            x = self.pool(x)

        x = self.bottleneck_3(x)
        skip_connections_3 = skip_connections_3[::-1]


        # decoder3 part
        for idx in range(0, len(self.ups_3), 2):
            x = self.ups_3[idx](x)
            skip_connection3 = skip_connections_3[idx // 2]

            if x.shape != skip_connection3.shape:
                x = TF.resize(x, size=skip_connection3.shape[2:])

            concat_skip = torch.cat((skip_connection3, x), dim=1)
            x = self.ups_3[idx + 1](concat_skip)

        return self.final_conv(x)

