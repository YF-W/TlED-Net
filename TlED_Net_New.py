import torch
from torch import nn
from torchvision import models as resnet_model
import torch.nn.functional as F
from thop import profile


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # [B, C//g, g, H, W]
        return x.reshape(B, C, H, W)

class Depthwise_Separable_conv(nn.Module):              #进行了通道混洗
    def __init__(self,in_channels,out_channels,dilation=1):
        super(Depthwise_Separable_conv, self).__init__()

        self.depth_conv=nn.Conv2d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=dilation,
                                    dilation=dilation,
                                    groups=in_channels)

        self.shuffle = ChannelShuffle(in_channels)

        self.point_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self,x):
        out = self.depth_conv(x)
        out = self.shuffle(out)
        out = self.point_conv(out)
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

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# Multi-branch Iterative Module
class MBIM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MBIM, self).__init__()

        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.branch_2 = SingleConv(in_channels, out_channels)

        self.branch_3_1 = nn.Sequential(
            nn.Conv2d(3*out_channels, out_channels, 5,1,2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.branch_3_2 = nn.Sequential(
            nn.Conv2d(4*out_channels, out_channels, 3,1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        b1 = self.branch_1(x)

        b2 = self.branch_2(x)

        b_c = torch.cat((b1,b2), dim=1)
        b3_1 = self.branch_3_1(torch.cat((b1+b2,b_c),dim=1))
        b3_2 = self.branch_3_2(torch.cat((b3_1, b1+b2, b_c),dim=1))

        return b3_2 + b1 + b2


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

        # rate = [[4, 6, 12], [6, 12, 18], [12, 18, 24]]
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
            nn.Conv2d(4 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):
        # rate = [[4, 6, 12], [6, 12, 18], [12, 18, 24]]

        a_1 = self.convs[1](x)
        a_1_1=torch.cat((a_1,x),dim=1)

        a_2 = self.convs[4](x)
        a_2_2=torch.cat((a_2,x),dim=1)

        # a_3 = self.convs[7](x)
        # a_3_3=torch.cat((a_3,x),dim=1)

        for i in range(1, 3):
            a_1=self.convs[1+i](a_1_1)
            a_1_1=torch.cat((a_1,a_1_1),dim=1)

            a_2=self.convs[4+i](a_2_2)
            a_2_2=torch.cat((a_2,a_2_2),dim=1)

            # a_3=self.convs[7+i](a_3_3)
            # a_3_3=torch.cat((a_3,a_3_3),dim=1)


        a_4 = self.convs[0](x)
        a_5 = self.convs[10](x)

        res = torch.cat((self.down_1(a_1_1),self.down_1(a_2_2), a_4, a_5), dim=1)
        out = self.relu(x + self.down_2(res))
        return out

class Conv_1x1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Conv_1x1, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)

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


class TransupConv_new(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride):
        super(TransupConv_new, self).__init__()
        self.trans=Conv_1x1(in_channels,out_channels)
        self.conv=nn.ConvTranspose2d(out_channels,out_channels,kernel_size=kernel_size,stride=stride)

    def forward(self,x):
        out = self.trans(x)
        out = self.conv(out)

        return out

class TlED_Net(nn.Module):
    def __init__(self,in_channels,out_channels, features=[[64, 128, 256],
                                                          [128,256, 512],
                                                          [128,256,512]]):
        super(TlED_Net, self).__init__()
        self.downs_1 = nn.ModuleList()
        self.downs_2 = nn.ModuleList()
        self.downs_3 = nn.ModuleList()

        self.ups_1 = nn.ModuleList()
        self.ups_2 = nn.ModuleList()
        self.ups_3 = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_2=nn.MaxPool2d(kernel_size=4,stride=4)
        self.pool_3=nn.MaxPool2d(kernel_size=8,stride=8)
        self.pool_4=nn.MaxPool2d(kernel_size=16,stride=16)

        resnet = resnet_model.resnet50(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.conv1_1x1s=nn.ModuleList()
        self.trans_conv=TransupConv_new(64,64,kernel_size=2,stride=2)
        self.relu=nn.ReLU(inplace=True)

        self.trans_conv_2=TransupConv_new(128,64,kernel_size=2,stride=2)
        self.trans_conv_2_2=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.conv1_1x1_2=Conv_1x1(256,512)

        self.trans_conv_3=TransupConv_new(1024,512,kernel_size=2,stride=2)
        self.trans_conv_4=TransupConv_new(1024,256,kernel_size=4,stride=4)
        self.trans_conv_5=TransupConv_new(1024,128,kernel_size=8,stride=8)
        self.trans_conv_6=TransupConv_new(1024,64,kernel_size=16,stride=16)

        self.conv1_1x1s_2=nn.ModuleList()
        for i in [128,256,512]:
            self.conv1_1x1s_2.append(Conv_1x1(64,i))

        #resnet_channels_trans
        trans_channels=[128,256,512,1024]
        for trans_channel in trans_channels:
            self.conv1_1x1s.append(Conv_1x1(trans_channel*2,trans_channel))

        rate = [[4, 6, 12], [6, 12, 18], [12, 18, 24]]
        self.DAspp=DASPP(64,64,rate)

        self.CSs=nn.ModuleList()
        self.CSs.append(CSMAttention(6 * 64))
        self.CSs.append(CSMAttention(5 * 128))
        self.CSs.append(CSMAttention(5 * 256))
        self.CSs.append(CSMAttention(3 *512))

        #Down part1 of demo
        self.downs_1.append(MBIM(3,64))
        in_channels = 64

        for feature in features[0][1:]:
            self.downs_1.append(Depthwise_Separable_conv(in_channels,feature))
            in_channels=feature

        self.downs_2.append(MBIM(128,256))
        self.downs_2.append(Depthwise_Separable_conv(256,512))

        #Down part3 of demo
        in_channels=128
        for feature in features[2][1:2]:
            self.downs_3.append(MBIM(in_channels, feature))
            in_channels = feature

        #Up part1 of demo
        for feature in reversed(features[2][:2]):
            self.ups_1.append(
                nn.ConvTranspose2d(
                    feature*2,feature,kernel_size=2,stride=2
                )
            )
            self.ups_1.append(Depthwise_Separable_conv(feature*3,feature))

        #Up part2 of demo
        for feature in reversed(features[1]):
            self.ups_2.append(
                nn.ConvTranspose2d(
                    feature*2,feature,kernel_size=2,stride=2
                )
            )
            self.ups_2.append(Depthwise_Separable_conv(feature * 3, feature))

        #Up part3 of demo
        for feature in reversed(features[0]):
            self.ups_3.append(
                nn.ConvTranspose2d(
                    feature*2,feature,kernel_size=2,stride=2
                )
            )
            if feature!=64:
                self.ups_3.append(SingleConv(feature * 5, feature))
            else:
                self.ups_3.append(SingleConv(feature * 6, feature))

        self.bottleneck1=Depthwise_Separable_conv(256*2,512)
        self.bottleneck2=Depthwise_Separable_conv(2*512,2*512)
        self.bottleneck3=Depthwise_Separable_conv(3*512,512)

        self.final_conv=nn.Conv2d(features[0][0],out_channels,kernel_size=1)



    def forward(self,x):
        skip_connections = []
        skip_connections_ex=[]
        skip_top=[]
        skip_bottleneck=[]


        #decoder ex_part
        e0=self.firstconv(x)
        e0=self.firstbn(e0)
        e0=self.firstrelu(e0)
                 #64,256,256
        skip_connections_ex.append(self.trans_conv(e0))     #64,512,512



        e1=self.encoder1(e0)
                 #256,256,256->128,256,256
        skip_connections_ex.append((self.conv1_1x1s[0](e1)))
        e2=self.encoder2(e1)
                 #512,128,128->256,128,128
        skip_connections_ex.append((self.conv1_1x1s[1](e2)))
        e3=self.encoder3(e2)
                 #1024,64,64->512,64,64
        skip_connections_ex.append((self.conv1_1x1s[2](e3)))
        e4=self.encoder4(e3)
                   #2048,32,32->1024,32,32
        skip_connections_ex.append((self.conv1_1x1s[3](e4)))

        x = self.downs_1[0](x)
        x=self.relu(x+self.DAspp(x))
        skip_connections.append(x)
        x = self.pool(x)

        #encoder part1
        for i in range(1,len(self.downs_1)):
            x=self.downs_1[i](x)
            skip_connections.append(x)
            x=self.pool(x)

        x=self.bottleneck1(torch.cat((self.pool(skip_connections_ex[2]),x),dim=1))           #512

        skip_bottleneck.append(x)

        #decoder part1
        index_de_1=1
        for idx in range(0,len(self.ups_1),2):
            x=self.ups_1[idx](x)
            concat_skip=torch.cat((skip_connections_ex[::-1][idx//2+2],skip_connections[::-1][idx//2],x),dim=1)

            index_de_1-=1
            x=self.ups_1[idx+1](concat_skip)

        skip_connections.append(x)

        # encoder part2
        x2 = self.pool(x)
        x2 = self.downs_2[0](x2)
        skip_connections.append(x2)

        x2 = self.pool(x2)
        x2 = self.downs_2[1](x2)
        skip_connections.append(x2)
        x2=self.pool(x2)
        x2=self.bottleneck2(torch.cat((self.pool(skip_bottleneck[0]),x2),dim=1))
        x2=self.relu(x2+skip_connections_ex[4])

        skip_bottleneck.append(self.trans_conv_3(x2))       #512        64
        skip_bottleneck.append(self.trans_conv_4(x2))       #256        128
        skip_bottleneck.append(self.trans_conv_5(x2))       #128        256
        skip_bottleneck.append(self.trans_conv_6(x2))       #64         512

        #decoder part2
        index_de_2=2
        for idx in range(0,len(self.ups_2),2):
            x2=self.ups_2[idx](x2)

            if idx==0:
                concat_skip=torch.cat((skip_connections_ex[3],skip_connections[::-1][idx//2],x2),dim=1)
            else:
                concat_skip=torch.cat((skip_connections_ex[::-1][idx//2+1],skip_connections[::-1][idx // 2],x2),dim=1)
            index_de_2-=1
            x2=self.ups_2[idx+1](concat_skip)

        x=x+x2
        skip_connections.append(x)

        #encoder part3
        for down in self.downs_3:   #256
            x=self.pool(x)
            x=down(x)
            skip_connections.append(x)
        x=self.pool(x)
        x=self.conv1_1x1_2(x)
        x=self.bottleneck3(self.CSs[3](torch.cat((skip_bottleneck[1],skip_connections_ex[3],x),dim=1)))

        for index in range(3,7,3):
            skip_top.append(self.trans_conv_2(skip_connections[index]))

        #decoder part3
        index_de_3=1
        for idx in range(0,len(self.ups_3),2):
            x=self.ups_3[idx](x)

            if idx==0:
                concat_skip=self.CSs[2](torch.cat((skip_bottleneck[2],skip_connections_ex[2],skip_connections[::-1][idx//2+5],skip_connections[::-1][idx // 2],x),dim=1))
            elif idx==2:
                concat_skip=self.CSs[1](torch.cat((skip_bottleneck[3],skip_connections_ex[1],skip_connections[::-1][idx//2+5],skip_connections[::-1][idx // 2],x),dim=1))
            else:
                concat_skip=self.CSs[0](torch.cat((skip_bottleneck[4],skip_top[1],skip_top[0],skip_connections_ex[0],skip_connections[0],x),dim=1))

            index_de_3-=1
            x=self.ups_3[idx+1](concat_skip)

        return self.final_conv(x)



# def test1():
#
#     # x4=torch.Tensor(2,3,224,224)
#     model=TlED_Net(in_channels=3,out_channels=1)
#
#     # model=Wei_Transformer_Block()
#     x = torch.Tensor(1, 3 , 224, 224)
#     flops, params = profile(model, inputs=(x,))
#     print(f'Flops: {flops}, params: {params}')
#
#     # preds=model(x4)
#     # print("shape of preds :",preds.shape)
#
#
# test1()