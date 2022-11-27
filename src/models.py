import math
from tokenize import group
import torch
import torch.nn as nn
import torch.nn.functional as F

class PUNet(nn.Module):

    def __init__(self, depth = [1, 1, 1, 1, 2]):
        super(PUNet, self).__init__()
    
        base_channel = 32

        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[LKB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel*2, base_channel*2, 3, 1),
            nn.Sequential(*[LKB(base_channel*2) for _ in range(depth[1])]),
            Down_scale(base_channel*2),
            BasicConv(base_channel*4, base_channel*4, 3, 1),
            nn.Sequential(*[LKB(base_channel*4) for _ in range(depth[2])]),
            Down_scale(base_channel*4),
        ])
        
        # Middle
        self.middle = nn.Sequential(*[LKB(base_channel*8) for _ in range(depth[3])])
        
        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale(base_channel*8),
            BasicConv(base_channel*8, base_channel*4, 3, 1),
            nn.Sequential(*[LKB(base_channel*4) for _ in range(depth[2])]),
            Up_scale(base_channel*4),
            BasicConv(base_channel*4, base_channel*2, 3, 1),
            nn.Sequential(*[LKB(base_channel*2) for _ in range(depth[1])]),
            Up_scale(base_channel*2),
            BasicConv(base_channel*2, base_channel, 3, 1),
            nn.Sequential(*[LKB(base_channel) for _ in range(depth[0])]),
        ])

        # conv
        self.conv_first = BasicConv(3, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 3, 3, 1, 1)
        self.refine = nn.Sequential(*[LKB(base_channel) for i in range(depth[4])])

    def skip_con(self, shortcuts_l, shortcuts_r):
        refine_l, refine_r = [], []
        for i in range(len(shortcuts_l)):
            shortcut_l, shortcut_r = shortcuts_l[i], shortcuts_r[i]
            # stereo fusion

            
            refine_l.append(shortcut_l)
            refine_r.append(shortcut_r)
        return refine_l, refine_r

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0 :
                shortcuts.append(x)

        return shortcuts, x
    
    def decoder(self, x, shortcuts):
        index = len(shortcuts)
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 ==0:
                index = index - 1
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)

        return x

    def to_one(self, x):
        return (torch.tanh(x) + 1) / 2
        
    def forward(self, x):
        x_l, x_r = x[:,:3,:,:], x[:,3:,:,:]
        x_l, x_r = self.conv_first(x_l), self.conv_first(x_r)
        shortcuts_l, x_l = self.encoder(x_l)
        shortcuts_r, x_r = self.encoder(x_r)
        x_l, x_r =  self.middle(x_l), self.middle(x_r)
        shortcuts_l, shortcuts_r = self.skip_con(shortcuts_l, shortcuts_r)
        x_l, x_r = self.decoder(x_l, shortcuts_l), self.decoder(x_r, shortcuts_r)
        x_l, x_r = self.refine(x_l), self.refine(x_r)
        x_l, x_r = self.conv_last(x_l), self.conv_last(x_r)
        x_l, x_r = self.to_one(x_l), self.to_one(x_r)
        return x_l, x_r

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, activation=True, transpose=False, groups=True):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            # layers.append(nn.LayerNorm(out_channel))
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class GroupConv(nn.Module):
    def __init__(self, dim, kernel_size, stride=1):
        super(GroupConv, self).__init__()
        padding = kernel_size // 2
        self.main = nn.Conv2d(dim, dim, kernel_size, padding=padding, stride=stride, bias=True, groups=dim)

    def forward(self, x):
        return self.main(x)

class MLP(nn.Module):
    def __init__(self, dim_in, din_out):
        super(MLP, self).__init__()
        self.main = nn.Conv2d(dim_in, din_out, 1, 1)

    def forward(self, x):
        return self.main(x)

class LKB(nn.Module):
    def __init__(self, dim):
        super(LKB, self).__init__()
        self.layer_1 = nn.Sequential(*[
            LayerNorm2d(dim),
            MLP(dim, dim*2),
            GroupConv(dim*2, 7),
            MLP(dim*2, dim)
        ])
        self.layer_2 = nn.Sequential(*[
            LayerNorm2d(dim),
            MLP(dim, dim*2),
            nn.GELU(),
            MLP(dim*2, dim)
        ])
        
    def forward(self, x):
        x = self.layer_1(x) + x
        x = self.layer_2(x) + x
        return x

class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)
        
    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x

class RDB(nn.Module):
    def __init__(self, dim, gd=32):
        super(RDB, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = BasicConv(dim, gd, 3, 1)
        self.conv2 = BasicConv(dim + gd, gd, 3, 1)
        self.conv3 = BasicConv(dim + gd * 2, gd, 3, 1)
        self.conv4 = BasicConv(dim + gd * 3, gd, 3, 1)
        self.conv5 = BasicConv(dim + gd * 4, dim, 3, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, dim, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = RDB(dim, gc)
        self.RDB2 = RDB(dim, gc)
        self.RDB3 = RDB(dim, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class CSI(nn.Module):
    def __init__(self, dim):
        # cross-scale interaraction
        super(CSI, self).__init__()
        self.low_up1 = nn.Sequential(*[Up_scale(dim*4), RRDB(dim*2)])
        self.low_up2 = nn.Sequential(*[Up_scale(dim*2), RRDB(dim)])

        self.mid_up = nn.Sequential(*[Up_scale(dim*2), RRDB(dim)])
        self.mid_down = nn.Sequential(*[Down_scale(dim*2), RRDB(dim*4)])

        self.high_down1 = nn.Sequential(*[Down_scale(dim), RRDB(dim*2)])
        self.high_down2 = nn.Sequential(*[Down_scale(dim*2), RRDB(dim*4)])

        self.conv_l = BasicConv(dim*12, dim*4, 1, 1)
        self.conv_m = BasicConv(dim*6, dim*2, 1, 1)
        self.conv_h = BasicConv(dim*3, dim, 1, 1)
        
    def forward(self, shortcuts):
        high, mid, low = shortcuts[0], shortcuts[1], shortcuts[2]
        
        l2m = self.low_up1(low)
        l2h = self.low_up2(l2m)
        m2h = self.mid_up(mid)
        m2l = self.mid_down(mid)
        h2m = self.high_down1(high)
        h2l = self.high_down2(h2m)

        low = self.conv_l(torch.cat([low, m2l, h2l], 1))
        mid = self.conv_m (torch.cat([l2m, mid, h2m], 1))
        high = self.conv_h(torch.cat([l2h, m2h, high], 1))

        return [high, mid, low]

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel*2, 3, 2)

    def forward(self, x):
        return self.main(x)

class Up_scale(nn.Module):
    def __init__(self, in_channel):
        super(Up_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel//2, kernel_size=4, activation=True, stride=2, transpose=True)

    def forward(self, x):
        return self.main(x)