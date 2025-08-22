import torch
from torch import nn
import functools
import torch.nn.functional as F
import numpy as np


class FlowUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d):
        super(FlowUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv, downnorm]
            up = [uprelu, upconv, upnorm]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.submodule = submodule
        self.predict_flow = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv2d(outer_nc, 2, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        if self.outermost:
            x_ = self.down(x)
            x_, x_pyramid, flow_pyramid = self.submodule(x_)
            x_ = self.up(x_)
            x_out = x_
        elif self.innermost:
            x_pyramid = []
            flow_pyramid = []
            x_ = self.up(self.down(x))
            x_out = torch.cat((x, x_), dim=1)
        else:
            x_ = self.down(x)
            x_, x_pyramid, flow_pyramid = self.submodule(x_)
            x_ = self.up(x_)
            x_out = torch.cat((x, x_), dim=1)

        flow = self.predict_flow(x_)
        x_pyramid = [x_] + x_pyramid
        flow_pyramid = [flow] + flow_pyramid
        return x_out, x_pyramid, flow_pyramid


class FlowUnet(nn.Module):
    def __init__(self, input_nc, nf=16, start_scale=2, num_scale=5, norm='batch', max_nf=512):
        super(FlowUnet, self).__init__()
        self.nf = nf
        self.norm = norm
        self.start_scale = 2
        self.num_scale = 5

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        conv_downsample = [
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=3, bias=use_bias),
            norm_layer(nf),
            nn.LeakyReLU(0.1)]
        nc = nf
        for i in range(int(np.log2(start_scale))):
            conv_downsample += [
                nn.Conv2d(nc, 2 * nc, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(2 * nc),
                nn.LeakyReLU(0.1)
            ]
            nc = nc * 2
        self.conv_downsample = nn.Sequential(*conv_downsample)

        unet_block = None
        for l in range(num_scale)[::-1]:
            outer_nc = min(max_nf, nc * 2 ** l)
            inner_nc = min(max_nf, nc * 2 ** (l + 1))
            innermost = (l == num_scale - 1)
            outermost = (l == 0)
            unet_block = FlowUnetSkipConnectionBlock(outer_nc, inner_nc, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer, innermost=innermost, outermost=outermost)

        self.unet_block = unet_block
        self.nf_out = min(max_nf, nc)
        self.predict_vis = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv2d(min(max_nf, nc), 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, pose1, pose2):
        input = torch.cat((pose1, pose2), dim=1)  # 对c通道进行cat
        x = self.conv_downsample(input)
        feat_out, x_pyr, flow_pyr = self.unet_block(x)
        vis = self.predict_vis(feat_out)

        flow_out = F.interpolate(flow_pyr[0], scale_factor=self.start_scale, mode='bilinear', align_corners=False)
        vis = F.interpolate(vis, scale_factor=self.start_scale, mode='bilinear', align_corners=False)
        return flow_out, vis, flow_pyr, feat_out


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False,
         norm_layer=nn.BatchNorm2d):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
        norm_layer(out_channels),
    )
    return model


def channel_mapping(in_channels, out_channels, norm_layer=nn.BatchNorm2d, bias=False):
    return conv(in_channels, out_channels, kernel_size=1, norm_layer=norm_layer, bias=bias)


class Identity(nn.Module):
    def __init__(self, dim=None):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class ResidualBlock(nn.Module):
    '''
    Derived from Variational UNet.
    '''

    def __init__(self, dim, dim_a, norm_layer=nn.BatchNorm2d, use_bias=False, activation=nn.ReLU(False),
                 use_dropout=False, no_end_norm=False):
        super(ResidualBlock, self).__init__()
        self.use_dropout = use_dropout
        self.activation = activation
        if dim_a is None:
            # w/o additional input
            if no_end_norm:
                self.conv = conv(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, norm_layer=Identity,
                                 bias=True)
            else:
                self.conv = conv(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, norm_layer=norm_layer,
                                 bias=use_bias)
        elif dim_a is not None:
            if dim_a <= 0:
                # w/o additional input
                if no_end_norm:
                    self.conv = conv(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, norm_layer=Identity,
                                     bias=True)
                else:
                    self.conv = conv(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, norm_layer=norm_layer,
                                     bias=use_bias)
        else:
            # w/ additional input
            self.conv_a = channel_mapping(in_channels=dim_a, out_channels=dim, norm_layer=norm_layer, bias=use_bias)
            if no_end_norm:
                self.conv = conv(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1, norm_layer=Identity,
                                 bias=True)
            else:
                self.conv = conv(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1, norm_layer=norm_layer,
                                 bias=use_bias)

    def forward(self, x, a=None):
        if a is None:
            # w/o additional input
            residual = x
        else:
            # w/ additional input
            a = self.conv_a(self.activation(a))
            residual = torch.cat((x, a), dim=1)
        residual = self.conv(self.activation(residual))
        out = x + residual
        if self.use_dropout:
            out = F.dropout(out, p=0.5, training=self.training)
        return out


class FlowUnet_v2(nn.Module):
    '''
    A variation of Unet that use residual blocks instead of convolution layer at each scale
    '''

    def __init__(self, input_nc, nf=64, max_nf=256, start_scale=2, num_scales=7, n_residual_blocks=2, norm='batch',
                 activation=nn.ReLU(False), use_dropout=False):
        super(FlowUnet_v2, self).__init__()
        self.input_nc = input_nc
        self.nf = nf
        self.max_nf = max_nf
        self.start_scale = start_scale
        self.num_scales = num_scales
        self.n_residual_blocks = n_residual_blocks
        self.norm = norm
        self.use_dropout = use_dropout

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()

        start_level = np.log2(start_scale).astype(np.int)
        pre_conv = [channel_mapping(input_nc, nf, norm_layer, use_bias)]
        for i in range(start_level):
            c_in = min(nf * (i + 1), max_nf)
            c_out = min(nf * (i + 2), max_nf)
            pre_conv += [
                ResidualBlock(c_in, None, norm_layer, use_bias, activation, use_dropout=use_dropout),
                activation,
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out)
            ]
        self.pre_conv = nn.Sequential(*pre_conv)

        for l in range(num_scales):
            c_in = min(nf * (start_level + l + 1), max_nf)
            c_out = min(nf * (start_level + l + 2), max_nf)
            # encoding layers
            for i in range(n_residual_blocks):
                self.__setattr__('enc_%d_res_%d' % (l, i),
                                 ResidualBlock(c_in, None, norm_layer, use_bias, activation, use_dropout=use_dropout))
            downsample = nn.Sequential(
                activation,
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out)
            )
            self.__setattr__('enc_%d_downsample' % l, downsample)
            # decoding layers
            upsample = nn.Sequential(
                activation,
                nn.Conv2d(c_out, c_in * 4, kernel_size=3, padding=1, bias=use_bias),
                nn.PixelShuffle(2),
                norm_layer(c_in)
            )
            self.__setattr__('dec_%d_upsample' % l, upsample)
            for i in range(n_residual_blocks):
                self.__setattr__('dec_%d_res_%d' % (l, i),
                                 ResidualBlock(c_in, c_in, norm_layer, use_bias, activation, use_dropout))
            # flow prediction
            pred_flow = nn.Sequential(
                activation,
                nn.Conv2d(c_in, 2, kernel_size=3, padding=1, bias=True)
            )
            self.__setattr__('pred_flow_%d' % l, pred_flow)
        # vis prediction
        self.pred_vis = nn.Sequential(
            activation,
            nn.Conv2d(nf * (1 + start_level), 3, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, pose1, pose2):
        x = torch.cat((pose1, pose2), dim=1)  # 对c通道进行cat

        hiddens = []
        flow_pyr = []
        x = self.pre_conv(x)
        # encode
        for l in range(self.num_scales):
            for i in range(self.n_residual_blocks):
                x = self.__getattr__('enc_%d_res_%d' % (l, i))(x)
                hiddens.append(x)
            x = self.__getattr__('enc_%d_downsample' % l)(x)
        # decode
        for l in range(self.num_scales - 1, -1, -1):
            x = self.__getattr__('dec_%d_upsample' % l)(x)
            for i in range(self.n_residual_blocks - 1, -1, -1):
                h = hiddens.pop()
                x = self.__getattr__('dec_%d_res_%d' % (l, i))(x, h)
            flow_pyr = [self.__getattr__('pred_flow_%d' % l)(x)] + flow_pyr

        feat_out = x
        flow_out = F.upsample(flow_pyr[0], scale_factor=self.start_scale, mode='bilinear', align_corners=False)
        vis_out = F.upsample(self.pred_vis(x), scale_factor=self.start_scale, mode='bilinear', align_corners=False)

        return flow_out, vis_out, flow_pyr, feat_out


model = FlowUnet(input_nc=36, nf=32, start_scale=2, num_scale=5, norm='batch')
