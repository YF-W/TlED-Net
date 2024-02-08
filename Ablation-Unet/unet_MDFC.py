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


class MDFC_4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDFC_4, self).__init__()

        self.brance_1=nn.Sequential(            
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.brance_2_2=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


        self.brance_3=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,7,1,3,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.brance_4=nn.Conv2d(in_channels,in_channels,kernel_size=1)


        self.channels_tran_1=nn.Conv2d(2*in_channels,in_channels,kernel_size=1)     #256->128
        self.channels_tran_2=nn.Conv2d(4*in_channels,in_channels,kernel_size=1)    #512->128
        self.channels_tran_3=nn.Conv2d(in_channels,out_channels,kernel_size=1)    #128->256



        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        b1=self.relu( x+self.channels_tran_1(self.brance_1(x)))         #1*
        b2=self.relu( x+self.channels_tran_2(torch.cat((self.brance_2_2(x),self.brance_2_2(x)),dim=1)))   #1*
        b3=self.relu( x+self.channels_tran_1(self.brance_3(x)))         #1*
        b4=self.brance_4(x)         #1*

        # b_ca=torch.cat(b1,b2,b3,b4,dim=1)
        b_cat=self.relu( x+self.channels_tran_2(torch.cat((b1,b2,b3,b4),dim=1)))
        return self.channels_tran_3(b_cat)



class MDFC_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDFC_3, self).__init__()

        self.branch_1=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


        self.branch_2=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.branch_3=nn.Conv2d(in_channels,in_channels,kernel_size=1)


        self.channel_tran_1=nn.Conv2d(2*in_channels,in_channels,kernel_size=1)
        self.channel_tran_2=nn.Conv2d(4*in_channels,in_channels,kernel_size=1)
        self.channel_tran_3=nn.Conv2d(3*in_channels,in_channels,kernel_size=1)
        self.channel_tran_4=nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.relu=nn.ReLU(inplace=True)


    def forward(self,x):
        b1=self.relu(x+self.channel_tran_1(self.branch_1(x)))       #1*
        b2=self.relu( x+self.channel_tran_2(torch.cat((self.branch_2(x),self.branch_2(x)),dim=1) )) #1*
        b3=self.branch_3(x)     #1*

        b_cat=self.relu( x+self.channel_tran_3(torch.cat((b1,b2,b3),dim=1)))

        return self.channel_tran_4(b_cat)





class UNET_MDFC(nn.Module):
    def __init__(
            self, in_channels, out_channels, features=[64, 128, 256, 512],
    ):
        super(UNET_MDFC, self).__init__()
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features[0:2]:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        self.downs.append(MDFC_4(128,256))
        self.downs.append(MDFC_3(256,512))

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

    def forward(self, x):
        skip_connections = []

        # decoder part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
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



