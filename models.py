import torch.nn as nn
import torch
import torch.nn.functional as F



class PDN_S(nn.Module):

    def __init__(self) -> None:
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
        self.conv4 = nn.Conv2d(256, 384, kernel_size=4, stride=1, padding=0)
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

    def __init__(self) -> None:
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
        self.conv5 = nn.Conv2d(512, 384, kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0)
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
    
class EncConv(nn.modules):

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

    def forward(self, x):
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        x = self.conv6(x)
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
        self.bilinear1 = nn.Upsample(scale_factor=3, mode='bilinear')
        self.bilinear2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bilinear3 = nn.Upsample(scale_factor=1.5, mode='bilinear')
        self.bilinear4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bilinear5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bilinear6 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bilinear7 = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.deconv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, 384, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)


    def forward(self, x):
        x = self.bilinear1(x)
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = self.bilinear2(x)
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = self.bilinear3(x)
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = self.bilinear4(x)
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = self.bilinear5(x)
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = self.bilinear6(x)
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = self.bilinear7(x)
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x
    

class AutoEncoder(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = EncConv()
        self.decoder = DecConv()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
class Teacher(nn.Module):

    def __init__(self,size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if size =='M':
            self.pdn = PDN_M()
        elif size =='S':
            self.pdn = PDN_S()
        

    def forward(self, x):
        x = self.pdn(x)
        return x
    

class Student(nn.Module):
    
    def __init__(self,size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ae = AutoEncoder()
        if size =='M':
            self.pdn = PDN_M()
        elif size =='S':
            self.pdn = PDN_S()
    def forward(self, x):
        pdn_out = self.pdn(x)
        ae_out = self.ae(x)
        return pdn_out,ae_out
    


