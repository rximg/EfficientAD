import torch
import torch.nn.functional as F
import os
#import WideResNet-101 from timm
# from timm.models.wide_resnet import WideResNet
import torchvision.models as models
from data_loader import ImageNetDataset
from torch.utils.data import Dataset, DataLoader
from models import Teacher
import pdb
from torchsummary import summary
import tqdm
from torch.optim.lr_scheduler import StepLR
from models import wide_resnet101_2
from torchvision import transforms
from itertools import cycle
# Require: A pretrained feature extractor Ψ : R 3×W×H → R 384×64×64.
# Require: A sequence of distillation training images Idist
# 1: Randomly initialize a teacher network T : R 3×256×256 → R 384×64×64 with an architecture as given in Table 5 or 6
# 2: for c ∈ 1, . . . , 384 do . Compute feature extractor channel normalization parameters µ Ψ ∈ R384 and σ Ψ ∈ R 384
# 3: Initialize an empty sequence X ← ( )
# 4: for iteration = 1, 2, . . . , 10 000 do
# 5: Choose a random training image Idist from Idist
# 6: Convert Idist to gray scale with a probability of 0.1
# 7: Compute I Ψ dist by resizing Idist to 3 × W × H using bilinear interpolation
# 8: Y Ψ ← Ψ(IΨ dist)
# 9: X ← X_ vec(Yc Ψ) . Append the channel output to X 10: end for 11: Set µ Ψ c to the mean and σc Ψ to the standard deviation of the elements of X
# 12: end for
# 13: Initialize the Adam [29] optimizer with a learning rate of 10−4 and a weight decay of 10−5 for the parameters of T
# 14: for iteration = 1, . . . , 60 000 do
# 15: Lbatch ← 0
# 16: for batch index = 1, . . . , 16 do
# 17: Choose a random training image Idist from Idist
# 18: Convert Idist to gray scale with a probability of 0.1
# 19: Compute IΨdist by resizing Idist to 3 × W × H using bilinear interpolation
# 20: Compute I0dist by resizing Idist to 3 × 256 × 256 using bilinear interpolation
# 21: YΨ ← Ψ(IΨdist)
# 22: Compute the normalized features Yˆ Ψ given by Yˆ Ψc = (YcΨ − µΨc)(σcΨ)−1for each c ∈ {1, . . . , 384}
# 23: Y0 ← T(I0dist)
# 24: Compute the squared difference between Yˆ Ψ and Y0 for each tuple (c, w, h) as Ddistc,w,h = (Yˆ Ψc,w,h − Y0c,w,h)2
# 25: Compute the loss Ldist as the mean of all elements Ddistc,w,h of Ddist
# 26: Lbatch ← Lbatch + Ldist
# 27: end for
# 28: Lbatch ← 16−1Lbatch
# 29: Update the parameters of T, denoted by θ, using the gradient ∇θLbatch
# 30: end for
# 31: return T


class DistillationTraining(object):

    def __init__(self,imagenet_dir,channel_size,batch_size,save_path,normalize_iter,train_iter=60000,resize=512,model_size='S', 
                wide_resnet_101_arch="Wide_ResNet101_2_Weights.IMAGENET1K_V2", print_freq=25) -> None:
        self.channel_size = channel_size
        self.mean = torch.empty(channel_size)
        self.std = torch.empty(channel_size)
        self.save_path = save_path
        self.imagenet_dir = imagenet_dir
        self.train_iter = train_iter
        self.model_size = model_size
        self.batch_size = batch_size
        self.normalize_iter = normalize_iter
        self.wide_resnet_101_arch = wide_resnet_101_arch
        self.print_freq = print_freq
        self.data_transforms = transforms.Compose([
                        transforms.Resize((resize, resize),),
                        transforms.RandomGrayscale(p=0.1), #6: Convert Idist to gray scale with a probability of 0.1 and 18: Convert Idist to gray scale with a probability of 0.1
                        transforms.ToTensor(),
                        ])

    def global_channel_normalize(self,dataloader):
        iterator = iter(dataloader)
        # for c in range(self.channel_size):
        # x_mean = torch.empty(0)
        # x_std = torch.empty(0)
        x = torch.empty(0)
        for iteration in tqdm.tqdm(range(self.normalize_iter)):
            ldist = next(iterator)[0]
            ldist = ldist.cuda()
            y = self.pretrain(ldist).detach().cpu()
            x = torch.cat((x,y),dim=0)
        self.mean = x.mean(dim=[0,2,3],keepdim=True).cuda()
        self.std = x.std(dim=[0,2,3],keepdim=True).cuda()

    def load_pretrain(self):
        self.pretrain = wide_resnet101_2(self.wide_resnet_101_arch, pretrained=True)
        # self.pretrain.load_state_dict(torch.load('pretrained_model.pth'))
        self.pretrain.eval()
        self.pretrain = self.pretrain.cuda()
        # print(summary(self.pretrain, (3, 512, 512)))
    
    def compute_mse_loss(self,teacher,ldist):
        y = self.pretrain(ldist)#torch.Size([8, 384, 64, 64])
        y = (y - self.mean)/self.std
        ldistresize = F.interpolate(ldist, size=(256, 256), mode='bilinear', align_corners=False)
        y0 = teacher(ldistresize)
        loss = F.mse_loss(y,y0)
        return loss

    def train(self,):
        self.load_pretrain()
        imagenet_dataset = ImageNetDataset(self.imagenet_dir, self.data_transforms)
        dataloader = DataLoader(imagenet_dataset, batch_size=self.batch_size, shuffle=True)
        iterator = cycle(iter(dataloader))
        teacher = Teacher(self.model_size)
        teacher = teacher.cuda()
        mean_param_path = '{}/imagenet_channel_std.pth'.format(self.save_path)
        if os.path.exists(mean_param_path):
            mean_param = torch.load(mean_param_path)
            self.mean = mean_param['mean'].cuda()
            self.std = mean_param['std'].cuda()
        else:
            self.global_channel_normalize(dataloader)
            torch.save({
                'mean': self.mean,
                'std': self.std
            }, '{}/imagenet_channel_std.pth'.format(self.save_path))
        optimizer = torch.optim.Adam(teacher.parameters(), lr=0.0001, weight_decay=0.00001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1, last_epoch=int(self.train_iter*0.9))
        best_loss = 1000
        loss_accum = 0
        for iteration in range(self.train_iter):
            ldist = next(iterator)[0]
            ldist = ldist.cuda()
            optimizer.zero_grad()
            loss = self.compute_mse_loss(teacher,ldist)
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()
            # scheduler.step()
            if (iteration+1) % self.print_freq == 0:
                loss_mean = loss_accum/self.print_freq
                print('iter:{},loss:{}'.format(iteration, loss_mean))
                if loss_mean < best_loss or best_loss == 1000:
                    best_loss = loss_mean
                    # save teacher
                    print('save best teacher at loss {}'.format(best_loss))
                    torch.save(teacher.state_dict(), '{}/best_teacher.pth'.format(self.save_path))
                loss_accum = 0

        # save teacher
        torch.save(teacher.state_dict(), '{}/last_teacher.pth'.format(self.save_path))
        
        

if __name__ == '__main__':
    imagenet_dir = './data/ImageNet'
    channel_size = 384
    save_path = './ckpt'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    distillation_training = DistillationTraining(
        imagenet_dir,channel_size,4,save_path,
        normalize_iter=500,train_iter=60000, 
        wide_resnet_101_arch="Wide_ResNet101_2_Weights.IMAGENET1K_V2")
    distillation_training.train()