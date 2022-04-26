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
#      self-attention
# ----------------------------------------
class MultiHeadDense(nn.Module):
    def __init__(self, d):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
    
    def forward(self, x):
        # x:[b, h*w, d]
        # x = torch.bmm(x, self.weight)
        x = F.linear(x, self.weight)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
    
    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()

class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel)
        self.key = MultiHeadDense(channel)
        self.value = MultiHeadDense(channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.positional_encoding_2d(c, h, w)
        x = x + pe
        x = x.reshape(b, c, h*w).permute(0, 2, 1) #[b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(c))#[b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        return x
    
class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, channelY, channelS):
        super(MultiHeadCrossAttention, self).__init__()
        self.Sconv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True)
        )
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True)
        )
        self.query = MultiHeadDense(channelS)
        self.key = MultiHeadDense(channelS)
        self.value = MultiHeadDense(channelS)
        self.conv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size()
        Yb, Yc, Yh, Yw = Y.size()
        Spe = self.positional_encoding_2d(Sc, Sh, Sw)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh*Yw).permute(0, 2, 1)
        V = self.value(S1)
        Ype = self.positional_encoding_2d(Yc, Yh, Yw)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh*Yw).permute(0, 2, 1)
        Y2 = self.Yconv2(Y)
        Q = self.query(Y1)
        K = self.key(Y1)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Sc))
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)
        Z = self.conv(x)
        Z = Z * S
        Z = torch.cat([Z, Y2], dim=1)
        return Z

class TransformerUp(nn.Module):
    def __init__(self, Ychannels, Schannels):
        super(TransformerUp, self).__init__()
        self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels)
        self.conv = nn.Sequential(
            nn.Conv2d(Ychannels, Schannels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(Schannels, Schannels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(Schannels, Schannels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, Y, S):
        x = self.MHCA(Y, S)
        x = self.conv(x)
        return x
# ----------------------------------------
#      Kernel Prediction Network (KPN)
# ----------------------------------------
class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.tconv = nn.Sequential(
            nn.MaxPool2d(2),
            TripleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.tconv(x)

class KPN(nn.Module):
    def __init__(self, color=True, burst_length=1, blind_est=True, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(KPN, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length+1)
        out_channel = (3 if color else 1) * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        
        # 各个卷积层定义
        # down = maxpooling+3*conv
        self.inc = TripleConv(in_channel, 64)
        self.conv1 = Down(64, 128)
        self.conv2 = Down(128, 256)
        self.conv3 = Down(256, 512)
        self.conv4 = Down(512, 512)
        self.MHSA = MultiHeadSelfAttention(512)  #self-attention
        # 6~8层要先上采样再卷积
        self.conv6 = TripleConv(512+512, 512)
        self.conv7 = TripleConv(256+512, 256)
        self.conv8 = TripleConv(256+128, out_channel)
        self.outc = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.fm = nn.Conv2d(out_channel, 1, 3, 1, 1)  #純看feature map
        
        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)
        
        self.conv_final = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

    # 前向传播函数
    def forward(self, data_x, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        inc = self.inc(data_x)
        conv1 = self.conv1(inc)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        sa = self.MHSA(conv4)  #self-attention
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv3, F.interpolate(sa, scale_factor=2, mode=self.upMode)], dim=1))
        conv7 = self.conv7(torch.cat([conv2, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1))
        conv8 = self.conv8(torch.cat([conv1, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        # return channel K*K*N
        core = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode))
        feature_map = self.fm(core)  #純看feature map

        pred1 = self.kernel_pred(data, core, white_level, rate=1)
        pred2 = self.kernel_pred(data, core, white_level, rate=2)
        pred3 = self.kernel_pred(data, core, white_level, rate=3)
        pred4 = self.kernel_pred(data, core, white_level, rate=4)

        pred_cat = torch.cat([torch.cat([torch.cat([pred1, pred2], dim=1), pred3], dim=1), pred4], dim=1)
        
        pred = self.conv_final(pred_cat)
        
        #pred = self.kernel_pred(data, core, white_level, rate=1)
        
        return pred, feature_map

class KernelConv(nn.Module):
    """
    the class of computing prediction
    """
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, 3, height, width)
            core_out[K] = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0, rate=1):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, color(1), height, width]
        :param core: [batch_size, N, dict(kernel), color(1), height, width]
        :return:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            #in here
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            #in here
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        #print('第一張圖之第1channel對應之9個feature_map')
        #print(core[3][0][0])
        
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):  #K=3, index沒用到
            if not img_stack:
                #in here
                padding_num = (K//2) * rate  #padding=1
                frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])  #雨圖邊緣補0
                for i in range(0, K):  # i: 0~2, j:0~2
                    for j in range(0, K):
                        img_stack.append(frame_pad[..., i*rate:i*rate + height, j*rate:j*rate + width])
                img_stack = torch.stack(img_stack, dim=2)
  
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            #print('img_stack:', img_stack.size())  #image_stack = frame原圖轉的 [batch, 3, 9, 1, 256, 256]
            pred_img.append(torch.sum(
                core[K].mul(img_stack), dim=2, keepdim=False
            ))
        pred_img = torch.stack(pred_img, dim=0)  #轉成tensor處理而已
        #print('pred_stack:', pred_img.size())  #[1, batch, 3, 1, 256, 256]
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
        #print("pred_img_i", pred_img_i.size())  #[batch, 3, 1, 256, 256]
        pred_img_i = pred_img_i.squeeze(2)
        #print("pred_img_i", pred_img_i.size())  #[batch, 3, 256, 256]
        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias
        # print('white_level', white_level.size())
        pred_img_i = pred_img_i / white_level
        #pred_img = torch.mean(pred_img_i, dim=1, keepdim=True)
        # print('pred_img:', pred_img.size())
        # print('pred_img_i:', pred_img_i.size())
        return pred_img_i

class LossFunc(nn.Module):
    """
    loss function of KPN
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
    
    kpn = KPN().cuda()
    a = torch.randn(4, 3, 224, 224).cuda()
    b = kpn(a, a)
    print(b.shape)
