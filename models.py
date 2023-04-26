import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torchvision.models.resnet import ResNet, Bottleneck
import torchvision
import pdb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def imagenet_norm_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    x_norm = (x - mean) / (std + 1e-11)
    return x_norm

class WideResNet(ResNet):


    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,target_dim=384):
        super(WideResNet, self).__init__(block, layers, num_classes, zero_init_residual,
                 groups, width_per_group, replace_stride_with_dilation,
                 norm_layer)
        self.target_dim = target_dim
        
    def _forward_impl(self, x):
        x = imagenet_norm_batch(x) #Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x0 = self.layer1(x)
        x1 = self.layer2(x0)
        x2 = self.layer3(x1)
        # pdb.set_trace()
        ret = self._proj(x1,x2)
        # x3 = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return ret
    
    
    def _proj(self,x1,x2):
        # [2, 512, 64, 64]->[2, 512, 64, 64],[2, 1024, 32, 32]->[2, 1024, 64, 64]
        # cat [2, 512, 64, 64],[2, 1024, 64, 64]->[2, 1536, 64, 64]
        # pool [2, 1536, 64, 64]->[2, 384, 32, 32]
        b,c,h,w = x1.shape
        x2 = F.interpolate(x2, size=(h,w), mode="bilinear", align_corners=False)
        features = torch.cat([x1,x2],dim=1)
        b,c,h,w = features.shape
        features = features.reshape(b,c,h*w)
        features = features.transpose(1,2)
        target_features = F.adaptive_avg_pool1d(features, self.target_dim)
        # pdb.set_trace()
        target_features = target_features.transpose(1,2)
        target_features = target_features.reshape(b,self.target_dim,h,w)
        return target_features


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = WideResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(url, progress=progress)
        model.load_state_dict(state_dict)
    return model

def wide_resnet101_2(arch, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    url = torchvision.models.get_weight(arch).url
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

# print(summary(wide_resnet101_2().cuda(), (3, 512, 512)))

class PDN_S(nn.Module):

    def __init__(self, last_kernel_size=384) -> None:
        super().__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # Conv-1 1×1 4×4 128 3 ReLU
        # AvgPool-1 2×2 2×2 128 1 -
        # Conv-2 1×1 4×4 256 3 ReLU
        # AvgPool-2 2×2 2×2 256 1 -
        # Conv-3 1×1 3×3 256 1 ReLU
        # Conv-4 1×1 4×4 384 0 -

        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x
    
class PDN_M(nn.Module):

    def __init__(self, last_kernel_size=384) -> None:
        super().__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # Conv-1 1×1 4×4 256 3 ReLU
        # AvgPool-1 2×2 2×2 256 1 -
        # Conv-2 1×1 4×4 512 3 ReLU
        # AvgPool-2 2×2 2×2 512 1 -
        # Conv-3 1×1 1×1 512 0 ReLU
        # Conv-4 1×1 3×3 512 1 ReLU
        # Conv-5 1×1 4×4 384 0 ReLU
        # Conv-6 1×1 1×1 384 0 -
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv2d(last_kernel_size, last_kernel_size, kernel_size=1, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x
    
class EncConv(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # EncConv-1 2×2 4×4 32 1 ReLU
        # EncConv-2 2×2 4×4 32 1 ReLU
        # EncConv-3 2×2 4×4 64 1 ReLU
        # EncConv-4 2×2 4×4 64 1 ReLU
        # EncConv-5 2×2 4×4 64 1 ReLU
        # EncConv-6 1×1 8×8 64 0 -
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)
        self.apply(weights_init)

    def forward(self, x):
        # pdb.set_trace()
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        x = self.enconv6(x)
        return x
    
class DecBlock(nn.Module):

    def __init__(self,scale_factor,stride,kernel_size,num_kernels,padding,activation,dropout_rate,):
        super().__init__()
        self.activation = activation    
        # self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.deconv = nn.Conv2d(num_kernels, num_kernels, kernel_size, stride, padding)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        
    def forward(self, x):
        x = self.upsample(x)
        x = F.relu(self.deconv(x))
        x = self.dropout(x)
        return x

class DecConv(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Bilinear-1 Resizes the 1×1 input features maps to 3×3
        # DecConv-1 1×1 4×4 64 2 ReLU
        # Dropout-1 Dropout rate = 0.2
        # Bilinear-2 Resizes the 4×4 input features maps to 8×8
        # DecConv-2 1×1 4×4 64 2 ReLU
        # Dropout-2 Dropout rate = 0.2
        # Bilinear-3 Resizes the 9×9 input features maps to 15×15
        # DecConv-3 1×1 4×4 64 2 ReLU
        # Dropout-3 Dropout rate = 0.2
        # Bilinear-4 Resizes the 16×16 input features maps to 32×32
        # DecConv-4 1×1 4×4 64 2 ReLU
        # Dropout-4 Dropout rate = 0.2
        # Bilinear-5 Resizes the 33×33 input features maps to 63×63
        # DecConv-5 1×1 4×4 64 2 ReLU
        # Dropout-5 Dropout rate = 0.2
        # Bilinear-6 Resizes the 64×64 input features maps to 127×127
        # DecConv-6 1×1 4×4 64 2 ReLU
        # Dropout-6 Dropout rate = 0.2
        # Bilinear-7 Resizes the 128×128 input features maps to 64×64
        # DecConv-7 1×1 3×3 64 1 ReLU
        # DecConv-8 1×1 3×3 384 1 -
        # self.bilinear1 = nn.Upsample(scale_factor=3, mode='bilinear')
        # self.bilinear2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.bilinear3 = nn.Upsample(scale_factor=1.7, mode='bilinear')
        # self.bilinear4 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.bilinear5 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.bilinear6 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.bilinear7 = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, 384, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)
        self.apply(weights_init)


    def forward(self, x):
        # x = self.bilinear1(x)
        x = F.interpolate(x, size=3, mode='bilinear')
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=8, mode='bilinear')
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=15, mode='bilinear')
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=32, mode='bilinear')
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=63, mode='bilinear')
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=127, mode='bilinear')
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=64, mode='bilinear')
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = EncConv()
        self.decoder = DecConv()

    def forward(self, x):
        x_norm = x = imagenet_norm_batch(x) #Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
class Teacher(nn.Module):
    def __init__(self,size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if size =='M':
            self.pdn = PDN_M(last_kernel_size=384)
        elif size =='S':
            self.pdn = PDN_S(last_kernel_size=384)
        self.pdn.apply(weights_init)

    def forward(self, x):
        x = imagenet_norm_batch(x) #Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        x = self.pdn(x)
        return x
    

class Student(nn.Module):
    def __init__(self,size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if size =='M':
            self.pdn = PDN_M(last_kernel_size=768) #The student network has the same architecture,but 768 kernels instead of 384 in the Conv-5 and Conv-6 layers.
        elif size =='S':
            self.pdn = PDN_S(last_kernel_size=768) #The student network has the same architecture, but 768 kernels instead of 384 in the Conv-4 layer
        self.pdn.apply(weights_init)

    def forward(self, x):
        x = imagenet_norm_batch(x) #Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        pdn_out = self.pdn(x)
        return pdn_out
    

if __name__ == '__main__':
    from torchsummary import summary
    import torch
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder()
    model = model.to('cuda')
    summary(model, (3, 256, 256))
