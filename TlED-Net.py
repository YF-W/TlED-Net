# Our code can be obtained at: https://github.com/YF-W/TlED-Net


import torch
from torch import nn
# import torchvision.transforms.functional as TF
from torchvision import models as resnet_model
import torch.nn.functional as F
# from Time import Timer










#k=3，放在encoder3部分以及decoder部分
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


#k=拓展卷积，放在encoder2部分

#4分支
class Threebranches_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Threebranches_conv, self).__init__()

        self.brance_1=nn.Sequential(            #两个连续卷积k=3
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


#3分支
class Twobranches_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Twobranches_conv, self).__init__()

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



# ASPP
# 两部分：asppconv、aspppooling

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_reates):
        super(ASPP, self).__init__()
        models = []
        models.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))


        for rate_item in atrous_reates:  # atrous conv部分
            for i in range(0,3):
                models.append(ASPPConv((i+1)*in_channels,out_channels,rate_item[i]))

        models.append(ASPPPooling(in_channels, out_channels))

        # 共11部分

        self.convs = nn.ModuleList(models)
        # print("len:",len(self.convs))
        # 降维
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
        # print(a_1_1.shape)

        a_2 = self.convs[4](x)
        a_2_2=torch.cat((a_2,x),dim=1)

        a_3 = self.convs[7](x)
        a_3_3=torch.cat((a_3,x),dim=1)

        # print(self.convs[2](a_1_1).shape)


        for i in range(1, 3):
            a_1=self.convs[1+i](a_1_1)
            a_1_1=torch.cat((a_1,a_1_1),dim=1)

            a_2=self.convs[4+i](a_2_2)
            a_2_2=torch.cat((a_2,a_2_2),dim=1)

            a_3=self.convs[7+i](a_3_3)
            a_3_3=torch.cat((a_3,a_3_3),dim=1)


        a_4 = self.convs[0](x)
        a_5 = self.convs[10](x)

        #  4 6 12、6、12、18、12、18、24

        res = torch.cat((self.down_1(a_1_1),self.down_1(a_2_2),self.down_1(a_3_3), a_4, a_5), dim=1)  # 通道维度连结
        out = self.relu(x + self.down_2(res))  # 残差
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




#CBAM——CAM
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // 16, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

#CBAM——SAM
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self,in_channels):
        super(CBAM, self).__init__()
        self.CAM=ChannelAttention(in_channels)
        self.SAM=SpatialAttention()

    def forward(self,x):
        out=self.CAM(x)*x
        out=self.SAM(out)*out
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(2*in_channels, 2*in_channels // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(2*in_channels // 16, in_channels, 1, bias=False))
        self.fc_2 = nn.Sequential(nn.Conv2d(2*in_channels, 2*in_channels // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(2*in_channels // 16, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()



    def _process_1(self,x,eps=1e-5):      #计算方差
        N, C, _, _ = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)[:,:,None]	#全局平均池化
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps	#全局方差
        channel_std = channel_var.sqrt()[:,:,None]		#开方求标准差

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



        #resnet-50老版本暂时能用
        resnet = resnet_model.resnet50(pretrained=True)

        #v0.13版本以后更新的
        # resnet=resnet_model.resnet50(weights=resnet_model.ResNet50_Weights.DEFAULT)


        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.conv1_1x1s=nn.ModuleList()
        self.trans_conv=nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.relu=nn.ReLU(inplace=True)

        self.trans_conv_2=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.trans_conv_2_2=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.conv1_1x1_2=Conv_1x1(256,512)


        self.trans_conv_3=nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.trans_conv_4=nn.ConvTranspose2d(1024,256,kernel_size=4,stride=4)
        self.trans_conv_5=nn.ConvTranspose2d(1024,128,kernel_size=8,stride=8)
        self.trans_conv_6=nn.ConvTranspose2d(1024,64,kernel_size=16,stride=16)



        self.conv1_1x1s_2=nn.ModuleList()
        for i in [128,256,512]:
            self.conv1_1x1s_2.append(Conv_1x1(64,i))





        #resnet_channels_trans
        trans_channels=[128,256,512,1024]
        for trans_channel in trans_channels:
            self.conv1_1x1s.append(Conv_1x1(trans_channel*2,trans_channel))

        rate = [[4, 6, 12], [6, 12, 18], [12, 18, 24]]

        #aspp
        self.Aspp=ASPP(512,512,rate)


        # self.CBAMS=nn.ModuleList()
        # for c in [64,128,256]:
        #     self.CBAMS.append(CBAM(7*c))
        # self.CBAMS.append(CBAM(6*512))

        self.CAs=nn.ModuleList()
        self.CAs.append(ChannelAttention(6 * 64))
        self.CAs.append(ChannelAttention(5 * 128))
        self.CAs.append(ChannelAttention(5 * 256))
        self.CAs.append(ChannelAttention(3 *512))


        #Down part1 of demo
        #通道大小依次为64,128,256
        for feature in features[0]:
            self.downs_1.append(DoubleConv(in_channels,feature))
            in_channels=feature




        self.downs_2.append(Threebranches_conv(128,256))
        self.downs_2.append(Twobranches_conv(256,512))




        #Down part3 of demo
        #通道大小依次为256,512
        in_channels=128
        for feature in features[2][1:2]:
            self.downs_3.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Up part1 of demo
        #通道大小依次为128,256
        for feature in reversed(features[2][:2]):
            self.ups_1.append(
                nn.ConvTranspose2d(
                    feature*2,feature,kernel_size=2,stride=2
                )
            )
            self.ups_1.append(DoubleConv(feature*3,feature))

        #Up part2 of demo
        for feature in reversed(features[1]):
            self.ups_2.append(
                nn.ConvTranspose2d(
                    feature*2,feature,kernel_size=2,stride=2        #s*n+k-2p-s=2*n+2-0-2=2*n
                )
            )
            self.ups_2.append(DoubleConv(feature * 3, feature))
            # if feature==512:
            #     self.ups_2.append(DoubleConv(feature*3,feature))
            # else:
            #     self.ups_2.append(DoubleConv(feature*4,feature))

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




        self.bottleneck1=DoubleConv(256*2,512)
        self.bottleneck2=DoubleConv(2*512,2*512)
        self.bottleneck3=DoubleConv(3*512,512)

        self.final_conv=nn.Conv2d(features[0][0],out_channels,kernel_size=1)



    def forward(self,x):
        skip_connections = []       #基础部分的skip
        skip_connections_ex=[]      #res-50
        skip_top=[]                 #两个encoder部分的输出
        skip_bottleneck=[]          #bottleneck1部分的skip
        skip_origin=[]              #原始的feature，需要进行下采样的部分


        #decoder ex_part
        e0=self.firstconv(x)
        e0=self.firstbn(e0)
        e0=self.firstrelu(e0)
        # print(e0.shape)         #64,256,256
        skip_connections_ex.append(self.trans_conv(e0))     #64,512,512



        e1=self.encoder1(e0)
        # print(e1.shape)         #256,256,256->128,256,256
        skip_connections_ex.append((self.conv1_1x1s[0](e1)))
        e2=self.encoder2(e1)
        # print(e2.shape)         #512,128,128->256,128,128
        skip_connections_ex.append((self.conv1_1x1s[1](e2)))
        e3=self.encoder3(e2)
        # print(e3.shape)         #1024,64,64->512,64,64
        skip_connections_ex.append((self.conv1_1x1s[2](e3)))
        e4=self.encoder4(e3)
        # print(e4.shape)           #2048,32,32->1024,32,32
        skip_connections_ex.append((self.conv1_1x1s[3](e4)))







        #encoder part1
        for i in range(len(self.downs_1)):      #64 128 256     h、w//2**3
            x=self.downs_1[i](x)
            skip_connections.append(x)
            x=self.pool(x)

        # size：256、128、64、32
        # skip_origin.append(self.conv1_1x1s_2[0](self.pool(skip_connections[0])))
        # skip_origin.append(self.conv1_1x1s_2[1](self.pool_2(skip_connections[0])))
        # skip_origin.append(self.conv1_1x1s_2[2](self.pool_3(skip_connections[0])))
        # skip_origin.append(self.conv1_1x1s_2[1](self.pool_3(skip_connections[0])))
        # skip_origin.append(self.conv1_1x1s_2[2](self.pool_4(skip_connections[0])))


        '''
        torch.Size([4, 128, 256, 256])
        torch.Size([4, 256, 128, 128])
        torch.Size([4, 512, 64, 64])
        torch.Size([4, 256, 64, 64])
        torch.Size([4, 512, 32, 32])
        '''




        x=self.bottleneck1(torch.cat((self.pool(skip_connections_ex[2]),x),dim=1))           #512
        # skip_bottleneck.append(x)
        # print(x.shape)

        x=self.relu(x+self.Aspp(x))                #ASPP
        skip_bottleneck.append(x)



        #decoder part1
        index_de_1=1
        for idx in range(0,len(self.ups_1),2):
            x=self.ups_1[idx](x)                            #下标为0,2共2次上采样
            concat_skip=torch.cat((skip_connections_ex[::-1][idx//2+2],skip_connections[::-1][idx//2],x),dim=1)     #对应倒序后跳跃连结下标为0,1

            index_de_1-=1
            x=self.ups_1[idx+1](concat_skip)
        # print("峰1:",x.shape)
        #output:torch.Size([3, 128, 80, 80])
 
        skip_connections.append(x)      #共4部分



        # encoder part2
        x2 = self.pool(x)           #res处
        x2 = self.downs_2[0](x2)
        skip_connections.append(x2)

        x2 = self.pool(x2)
        x2 = self.downs_2[1](x2)
        skip_connections.append(x2)
        x2=self.pool(x2)



        # torch.Size([3, 512, 10, 10])



        x2=self.bottleneck2(torch.cat((self.pool(skip_bottleneck[0]),x2),dim=1))


        # x=x+skip_connections_ex[4]


        #来自renet-50 layer4部分的连结
        x2=self.relu(x2+skip_connections_ex[4])

        skip_bottleneck.append(self.trans_conv_3(x2))       #512        64
        skip_bottleneck.append(self.trans_conv_4(x2))       #256        128
        skip_bottleneck.append(self.trans_conv_5(x2))       #128        256
        skip_bottleneck.append(self.trans_conv_6(x2))       #64         512


        #共5个,0为bottleneck_1


        # x=self.relu(x+self.Aspp(x))                #ASPP


        # torch.Size([3, 1024, 10, 10])
        # skip_connections共6部分

        #decoder part2
        index_de_2=2
        for idx in range(0,len(self.ups_2),2):
            x2=self.ups_2[idx](x2)                            #下标为0,2,4进行上采样

            if idx==0:
                concat_skip=torch.cat((skip_connections_ex[3],skip_connections[::-1][idx//2],x2),dim=1)
            else:
                concat_skip=torch.cat((skip_connections_ex[::-1][idx//2+1],skip_connections[::-1][idx // 2],x2),dim=1)
            index_de_2-=1
            x2=self.ups_2[idx+1](concat_skip)
        # print("峰2:",x.shape)
        # torch.Size([3, 128, 80, 80])

        x=x+x2

        skip_connections.append(x)      #共7部分

        #encoder part3
        for down in self.downs_3:   #256
            x=self.pool(x)
            x=down(x)
            skip_connections.append(x)
        x=self.pool(x)
        #torch.Size([3, 256, 20, 20])
        # skip_connections共8部分

        x=self.conv1_1x1_2(x)       #转为512




        x=self.bottleneck3(self.CAs[3](torch.cat((skip_bottleneck[1],skip_connections_ex[3],x),dim=1)))
        # torch.Size([3, 512, 20, 20])


        #处理skip_top部分

        for index in range(3,7,3):
            skip_top.append(self.trans_conv_2(skip_connections[index]))
            # print(skip_top[int(index/3)-1].shape)

        #decoder part3
        index_de_3=1
        for idx in range(0,len(self.ups_3),2):
            x=self.ups_3[idx](x)            #上采样下标0,2,4

            if idx==0:
                concat_skip=self.CAs[2](torch.cat((skip_bottleneck[2],skip_connections_ex[2],skip_connections[::-1][idx//2+5],skip_connections[::-1][idx // 2],x),dim=1))
            elif idx==2:
                concat_skip=self.CAs[1](torch.cat((skip_bottleneck[3],skip_connections_ex[1],skip_connections[::-1][idx//2+5],skip_connections[::-1][idx // 2],x),dim=1))
            else:
                concat_skip=self.CAs[0](torch.cat((skip_bottleneck[4],skip_top[1],skip_top[0],skip_connections_ex[0],skip_connections[0],x),dim=1))

            index_de_3-=1
            x=self.ups_3[idx+1](concat_skip)
        # torch.Size([3, 64, 161, 161])

        return self.final_conv(x)



def test1():

    x4=torch.Tensor(4,3,512,512)       #需要改变输入通道数为3
    model=TlED_Net(in_channels=3,out_channels=1)

    preds=model(x4)
    print("shape of preds :",preds.shape)




if __name__=='__main__':
    # time1=Timer()
    # time1.start()
    test1()
    # time1.stop()
    # time1.ptime_second()
    # time1.ptime_min()
