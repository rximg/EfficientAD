import torch
import torch.nn.functional as F

#import WideResNet-101 from timm
from timm.models.wide_resnet import WideResNet
from data_loader import ImageNetDataset
from torch.utils.data import Dataset, DataLoader
from models import Teacher
# Require: A pretrained feature extractor Ψ : R
# 3×W×H → R
# 384×64×64
# .
# Require: A sequence of distillation training images Idist
# 1: Randomly initialize a teacher network T : R
# 3×256×256 → R
# 384×64×64 with an architecture as given in Table 5 or 6
# 2: for c ∈ 1, . . . , 384 do . Compute feature extractor channel normalization parameters µ
# Ψ ∈ R
# 384 and σ
# Ψ ∈ R
# 384
# 3: Initialize an empty sequence X ← ( )
# 4: for iteration = 1, 2, . . . , 10 000 do
# 5: Choose a random training image Idist from Idist
# 6: Convert Idist to gray scale with a probability of 0.1
# 7: Compute I
# Ψ
# dist by resizing Idist to 3 × W × H using bilinear interpolation
# 8: Y
# Ψ ← Ψ(I
# Ψ
# dist)
# 9: X ← X_ vec(Yc
# Ψ) . Append the channel output to X
# 10: end for
# 11: Set µ
# Ψ
# c
# to the mean and σc
# Ψ to the standard deviation of the elements of X
# 12: end for
# 13: Initialize the Adam [29] optimizer with a learning rate of 10−4
# and a weight decay of 10−5
# for the parameters of T
# 14: for iteration = 1, . . . , 60 000 do
# 15: Lbatch ← 0
# 16: for batch index = 1, . . . , 16 do
# 17: Choose a random training image Idist from Idist
# 18: Convert Idist to gray scale with a probability of 0.1
# 19: Compute I
# Ψ
# dist by resizing Idist to 3 × W × H using bilinear interpolation
# 20: Compute I
# 0dist by resizing Idist to 3 × 256 × 256 using bilinear interpolation
# 21: Y
# Ψ ← Ψ(I
# Ψ
# dist)
# 22: Compute the normalized features Yˆ Ψ given by Yˆ Ψ
# c = (Yc
# Ψ − µ
# Ψ
# c
# )(σc
# Ψ)
# −1
# for each c ∈ {1, . . . , 384}
# 23: Y
# 0 ← T(I
# 0dist)
# 24: Compute the squared difference between Yˆ Ψ and Y
# 0 for each tuple (c, w, h) as Ddist
# c,w,h = (Yˆ Ψ
# c,w,h − Y
# 0c,w,h)
# 2
# 25: Compute the loss Ldist as the mean of all elements Ddist
# c,w,h of Ddist
# 26: Lbatch ← Lbatch + Ldist
# 27: end for
# 28: Lbatch ← 16−1Lbatch
# 29: Update the parameters of T, denoted by θ, using the gradient ∇θLbatch
# 30: end for
# 31: return T


class DistillationTraining(object):

    def __init__(self,channel_size) -> None:
        self.channel_size = channel_size
        self.mean = torch.empty(channel_size)
        self.std = torch.empty(channel_size)

    def normalize_channel(self,dataloader):
        for c in range(self.channel_size):
            X = torch.empty(0)
            for iteration in range(10000):
                ldist = next(dataloader)
                ldist = ldist.cuda()
                y = self.pretrain(ldist)
                y = y[:,c,:,:]
                y = y.view(-1)
                X = torch.cat((X,y),0)
            mean = torch.mean(X)
            std = torch.std(X)
            self.mean[c] = mean
            self.std[c] = std

    def load_pretrain(self):
        self.pretrain = WideResNet(depth=101, num_classes=1000, widen_factor=2, drop_rate=0.3, drop_connect_rate=0.2)
        self.pretrain.load_state_dict(torch.load('pretrained_model.pth'))
        self.pretrain.eval()
        self.pretrain = self.pretrain.cuda()
    
    def compute_mse_loss(self,teacher,ldist):
        y = self.pretrain(ldist)
        y = y.view(y.shape[0],y.shape[1],-1)
        y = torch.transpose(y,1,2)
        y = y.view(-1,y.shape[2])
        y = (y - self.mean)/self.std
        y = y.view(y.shape[0],y.shape[1],1,1)
        y = torch.transpose(y,1,2)
        y = torch.transpose(y,2,3)
        y = y.view(y.shape[0],y.shape[1],y.shape[2],y.shape[3])
        y0 = teacher(ldist)
        loss = F.mse_loss(y,y0)
        return loss

    def train(self,):
        self.load_pretrain()
        imagenet_dataset = ImageNetDataset()
        dataloader = DataLoader(imagenet_dataset, batch_size=32, shuffle=True)
        ldist = next(dataloader)
        ldist = ldist.cuda()
        teacher = Teacher()
        teacher = teacher.cuda()
        self.normalize_channel(dataloader)
        optimizer = torch.optim.Adam(teacher.parameters(), lr=0.0001, weight_decay=0.00001)
        for iteration in range(60000):
            ldist = next(dataloader)
            ldist = ldist.cuda()
            optimizer.zero_grad()
            loss = self.compute_mse_loss(teacher,ldist)
            loss.backward()
            optimizer.step()
            if iteration% 10 ==0:
                print('iter:{},loss:{}'.format(iteration,loss.item()))
        
        