import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from spectral import SpectralNorm

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
    
# ----------------------------------------
#      Generator
# ----------------------------------------
class Generator (nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        out_channel = 64
        # down = maxpooling+3*conv
        self.inc = TipleConv(3, 64)
        self.conv1 = Down(64, 128)
        self.conv2 = Down(128, 256)
        self.conv3 = Down(256, 512)
        self.conv4 = Down(512, 512)
        self.FeatureSA = Self_Attn(512, 'relu')  #self-attention
        # up = deconv + 3*conv
        self.conv6 = Up(512, 512, 512)
        self.conv7 = Up(256, 512, 256)
        self.conv8 = Up(128, 256, out_channel)
        self.outc = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1),
            nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        )
        self.fm = nn.Conv2d(out_channel, 1, 3, 1, 1)  #feature map
        
        #Rain de-raining Module
        self.rd_conv1 = nn.Sequential(  #256
            nn.Conv2d(4, 64, 5, 1, 2),
            nn.ReLU()
            )
        self.rd_conv2 = nn.Sequential(  #down 128
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
            )
        self.rd_conv3 = nn.Sequential(  
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
            )
        self.rd_conv4 = nn.Sequential(  #down 64
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
            )
        self.rd_conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.rd_conv_down3 = nn.Sequential(  #down 32
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU()
            )
        self.rd_conv_down4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
            )
        self.rd_conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
            )
        self.DerainSA = Self_Attn(512, 'relu')  #self-attention
        self.diconv1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 2, dilation = 2),
            nn.ReLU()
            )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 4, dilation = 4),
            nn.ReLU()
            )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 8, dilation = 8),
            nn.ReLU()
            )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 16, dilation = 16),
            nn.ReLU()
            )
        self.rd_conv7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
            )
        self.rd_conv8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        self.rd_conv_up0 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        self.rd_conv9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        self.rd_conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.ReLU()
            )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
            )
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
            )

    def forward(self, Rainy_image_x, Rainy_image):
        #Rain Feature Extraction Module:
        #:param data_with_est: if not blind estimation, it is same as data
        #:param data:
        #:return: img_pred, img_featuremap
        #down-sampling
        inc = self.inc(Rainy_image_x)
        conv1 = self.conv1(inc)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        feature_sa = self.FeatureSA(conv4)  #self-attention
        #up-sampling + crop
        conv6 = self.conv6(conv3, feature_sa)
        conv7 = self.conv7(conv2, conv6)
        conv8 = self.conv8(conv1, conv7)
        # return channel K*K*N
        feature_map = self.outc(conv8)
        feature_map = self.fm(feature_map)  #feature map
        
        #Rain de-raining Module: skip-network
        x = torch.cat((Rainy_image, feature_map), 1)  #256
        x = self.rd_conv1(x)
        res1 = x
        x = self.rd_conv2(x)  #down 128
        x = self.rd_conv3(x)
        res2 = x
        x = self.rd_conv4(x)  #down 64
        x = self.rd_conv5(x)
        res3 = x
        x = self.rd_conv_down3(x)  #down 64 -> 32
        x = self.rd_conv_down4(x)  
        x = self.rd_conv6(x)
        x = self.diconv1(x)  #dilated conv
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.DerainSA(x)  #self-attention
        x = self.rd_conv7(x)
        x = self.rd_conv8(x)
        x = self.deconv0(x)  #up 32 -> 64
        x = x + res3
        x = self.rd_conv_up0(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)  #up 128
        x = x + res2
        x = self.rd_conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)  #up
        x = x + res1
        x = self.rd_conv10(x)
        pred = self.output(x)
        return pred, feature_map

class TipleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.MaxPool2d(2),
            TipleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, channels_x, channels_y, channels_out):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channels_y, channels_y, 4, 2, 1)
        self.tconv = TipleConv(channels_x+channels_y, channels_out)

    def forward(self, x, y):
        deconv_y = self.deconv(y)
        cont_xy = torch.cat([x, deconv_y], dim=1)
        dconvxy = self.tconv(cont_xy)
        return dconvxy

# ----------------------------------------
#      Self-attention (SAGAN ver.)
# ----------------------------------------
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        #return out,attention
        return out    
    
class LossFunc(nn.Module):
    """
    loss function of Generator
    """
    def __init__(self, coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        self.loss_basic = LossBasic(gradient_L1)
        self.loss_anneal = LossAnneal(alpha, beta)

    def forward(self, pred_img_i, pred_img, ground_truth, global_step):
        """
        forward function of loss_func
        :param frames: frame_1 ~ frame_N, shape: [batch, N, 3, height, width]
        :param core: a dict coverted by ......
        :param ground_truth: shape [batch, 3, height, width]
        :param global_step: int
        :return: loss
        """
        return self.coeff_basic * self.loss_basic(pred_img, ground_truth), self.coeff_anneal * self.loss_anneal(global_step, pred_img_i, ground_truth)

class LossBasic(nn.Module):
    """
    Basic loss function.
    """
    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        return self.l2_loss(pred, ground_truth) + \
               self.l1_loss(self.gradient(pred), self.gradient(ground_truth))

class LossAnneal(nn.Module):
    """
    anneal loss function
    """
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        """
        :param global_step: int
        :param pred_i: [batch_size, N, 3, height, width]
        :param ground_truth: [batch_size, 3, height, width]
        :return:
        """
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth)
        loss /= pred_i.size(1)
        return self.beta * self.alpha ** global_step * loss

class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )
        
if __name__ == '__main__':
    
    RFE = Rain_Feature_Extraction_Module().cuda()
    a = torch.randn(4, 3, 224, 224).cuda()
    b = RFE(a, a)
    print(b.shape)
