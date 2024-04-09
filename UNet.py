import torch
from torch import nn as nn
from torch.nn import functional as F
from functools import partial

class UNet(nn.Module):
    """
    f_maps:解码器的每一层的特征图的数量
    """
    def __init__(self, in_channels, out_channels, basic_module, f_maps, layer_order='gcr',
                 num_groups=8, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1):
        super(UNet, self).__init__()
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size)

        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups)

        # 最后一层1x1卷积将
        self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # 翻转顺序和decoder对齐
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        return x

def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                    pool_kernel_size):
    # [64,128,256,512,1024]
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            # apply conv_coord only in the first encoder if any
            encoder = Encoder(in_channels, out_feature_num,
                              apply_pooling=False, 
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding,
                              )
        else:
            encoder = Encoder(f_maps[i - 1], out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding,
                              )

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups):
    # [1024,512,256,128,64]
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding
                          )
        decoders.append(decoder)
    return nn.ModuleList(decoders)
class Encoder(nn.Module):
    """
    为每层加入池化
    """
    def __init__(self, in_channels, out_channels, basic_module, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max',  conv_layer_order='gcr',
                 num_groups=8, padding=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    
    """
    def __init__(self, in_channels, out_channels,basic_module, conv_kernel_size=3, scale_factor=(2, 2), 
                 conv_layer_order='gcr', num_groups=8, mode='nearest', padding=1):
        super(Decoder, self).__init__()
        if basic_module == DoubleConv:
            # 如果使用DoubleConv作为basic_module,则使用插值进行上采样,使用concat连接
            self.upsampling = InterpolateUpsampling(mode=mode)
            self.joining = partial(self._joining, concat=True)
        else:
            # 如果使用ResNetBlock作为basic_module,则使用转置卷积上采样,使用sum连接
            self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                      kernel_size=conv_kernel_size, scale_factor=scale_factor)
            self.joining = partial(self._joining, concat=False)
            # adapt the number of in_channels for the ResNetBlock
            in_channels = out_channels


        self.basic_module = basic_module(in_channels, out_channels,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding
                                         )

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x
class ResNetBlock(nn.Module):
    """
    可以用来替代DoubleConv
    同样
    默认'cge'作为SingleConv的顺序
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,padding=1, order='cge', num_groups=8):
        super(ResNetBlock, self).__init__()
        conv2_in_channels = conv1_out_channels = (in_channels+out_channels) // 2
        # 第一次卷积
        self.conv1 = SingleConv(in_channels, conv1_out_channels, kernel_size=kernel_size,padding=1, order=order, num_groups=num_groups)
        # 将第二次卷积中去掉非线性激活,移到残差连接之后
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv2 = SingleConv(conv2_in_channels, out_channels, kernel_size=kernel_size, padding=1,order=n_order,num_groups=num_groups)

        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)
        # 1x1的卷积调整通道数
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv3 = nn.Identity()

    def forward(self, X):
        residual = self.conv3(X)
        X = self.conv1(X)
        X = self.conv2(X)
        
        X += residual
        X = self.non_linearity(X)

        return X
class DoubleConv(nn.Sequential):
    """
    一个包含两个连续卷积的模块
    最好设定卷积层的padding,保证输入输出尺寸不变
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(DoubleConv, self).__init__()

        conv2_in_channels = conv1_out_channels = (in_channels+out_channels) // 2

        self.add_module('SingleConv1',
                        SingleConv(in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding))

        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, out_channels, kernel_size, order, num_groups,
                                   padding=padding))

class SingleConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1, is3d=True):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
            self.add_module(name, module)

def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
    """
    返回list of tuple (name, module),包含一层卷积,一层非线性激活和一层可选的批/组归一化
        'cr' -> conv + ReLU
        'gcr' -> groupnorm + conv + ReLU
        'cl' -> conv + LeakyReLU
        'ce' -> conv + ELU
        'bcr' -> batchnorm + conv + ReLU
    """
    assert 'c' in order, "必须有卷积层"
    assert order[0] not in 'rle', '非线性激活不能是任何层的第一步操作'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            modules.append(('conv', conv))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'输入的通道数需要能被num_groups整除. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')

            bn = nn.BatchNorm2d

            if is_before_conv:
                modules.append(('batchnorm', bn(in_channels)))
            else:
                modules.append(('batchnorm', bn(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l'', 'c'], 'e")

    return modules
    
class AbstractUpsampling(nn.Module):
    """
    上采样的抽象类,给定的实现应该使用插值或反卷积等在四维tensor输入上上采样
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)
class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True

    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2)):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                      padding=1)
        super().__init__(upsample)
