import torch
import torch.nn.functional as F
import random
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from models import Teacher,Student,AutoEncoder
from torch.optim.lr_scheduler import StepLR
from data_loader import TrainImageOnlyDataset,ImageNetDataset,MVTecDataset
import tqdm
import os.path as osp
import pdb
import os

from PIL import Image

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Reduced_Student_Teacher(object):


    def __init__(self,mvtech_dir,imagenet_dir,ckpt_path,model_size='S',batch_size=8,channel_size=384,resize=256,fmap_size  = 256) -> None:
        self.mvtech_dir = mvtech_dir
        self.imagenet_dir = imagenet_dir
        self.ckpt_path = ckpt_path
        self.student = Student(model_size)
        self.student = self.student.cuda()
        # self.student.apply(weights_init)
        self.teacher = Teacher(model_size)
        self.load_pretrain_teacher()
        self.ae = AutoEncoder()
        self.ae = self.ae.cuda()
        # self.ae.apply(weights_init)
        self.channel_size = channel_size
        self.fmap_size = (fmap_size,fmap_size)
        self.channel_mean,self.channel_std = None,None
        self.batch_size = batch_size
        self.data_transforms = transforms.Compose([
                        transforms.Resize((resize, resize), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        ])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((resize, resize)),
                        transforms.ToTensor()])

    def load_pretrain_teacher(self):
        self.teacher.load_state_dict(torch.load(self.ckpt_path+'/best_teacher.pth'))
        self.teacher.eval()
        self.teacher = self.teacher.cuda()
        print('load teacher model from {}'.format(self.ckpt_path+'/best_teacher.pth'))

    def global_channel_normalize(self,dataloader):
        # iterator = iter(dataloader)
        # for c in range(self.channel_size):
        # x_mean = torch.empty(0)
        # x_std = torch.empty(0)
        x = torch.empty(0)
        for item in tqdm.tqdm(dataloader):
            # pdb.set_trace()
            ldist = item['image'].cuda()
            # ldist = iteration['image'].cuda()
            y = self.teacher(ldist).detach().cpu()
            # ymean = y.mean(dim=[2,3],keepdim=True)
            # ystd = y.std(dim=[2,3],keepdim=True)
            x = torch.cat((x,y),0)
            # x_std = torch.cat((x_std,y),0)
        self.channel_mean = x.mean(dim=[0,2,3],keepdim=True).cuda()
        self.channel_std = x.std(dim=[0,2,3],keepdim=True).cuda()
        return self.channel_mean,self.channel_std

    def choose_random_aug_image(self,image):
        aug_index = random.choice([1,2,3])
        # Sample an augmentation coefficient λ from the uniform distribution U(0.8, 1.2)
        coefficient = random.uniform(0.8,1.2)
        if aug_index == 1:
            img_aug = transforms.functional.adjust_brightness(image,coefficient)
        elif aug_index == 2:
            img_aug = transforms.functional.adjust_contrast(image,coefficient)
        elif aug_index == 3:
            img_aug = transforms.functional.adjust_saturation(image,coefficient)
        return img_aug

    # def squard_diffence(self,Y0,Y1):

    def compute_normalize_teacher_out(self,t_out):
        normal_t_out = []
        # pdb.set_trace()
        for c in range(self.channel_size):
            c_out = t_out[:,c,:,:]
            normal_t_out.append((c_out-self.channel_mean[c])/self.channel_std[c])
        normal_t_out = torch.stack(normal_t_out,dim=1)
        return normal_t_out

    def loss_st(self,image,imagenet_iterator,teacher:Teacher,student:Student):
        #Choose a random training image_train from dataloader_rain
        # image = self.dataloader[0]
        #Forward pass of the student–teacher pair
        t_pdn_out = teacher(image)
        b,c,h,w = t_pdn_out.shape
        #Compute the normalized teacher output given by normal_t_out = (teacher_out_channel − mean_c)/std_c for each channcel ∈ {1, . . . , 384}
        
        # normal_t_out = self.compute_normalize_teacher_out(t_pdn_out)
        normal_t_out = (t_pdn_out-self.channel_mean)/self.channel_std
        s_pdn_out,s_ae_out = student(image)
        # Compute the squared difference between normal_t_out and s_pdn_out for each tuple (c, w, h) as DST c,w,h = (Yˆc,w,h − YSTc,w,h)2
        distance_s_t = torch.pow(normal_t_out-s_pdn_out,2)
        # pdb.set_trace()
        # Compute the 0.999-quantile of the elements of DST, denoted by dhard
        # dhard = torch.quantile(distance_s_t,0.999)
        dhard = np.percentile(distance_s_t.detach().cpu().numpy(), 99.9)
        # Compute the loss Lhard as the mean of all DST c,w,h ≥ dhard
        hard_data = distance_s_t[distance_s_t>=dhard]

        # pdb.set_trace() 
        Lhard = torch.mean(hard_data)
        # Choose a random pretraining image P ∈ R 3×256×256 from ImageNet [54]
        image_p = next(imagenet_iterator)
        # pdb.set_trace()
        s_imagenet_out,_ = student(image_p[0].cuda())
        # Compute the loss LST = Lhard + (384 · 64 · 64)−1 P 384 c=1 k S(P)ck 2 F
        N = (1/(c*h*w))*torch.sum(torch.pow(s_imagenet_out,2))
        # print('loss Lhard {}, loss N {}'.format(Lhard,N))
        loss_st = Lhard + N
        return loss_st
    
    def loss_ae(self,image,teacher:Teacher,student:Student,autoencoder:AutoEncoder):
        aug_img = self.choose_random_aug_image(image=image)
        aug_img = aug_img.cuda()
        ae_out = autoencoder(aug_img)
        t_out = teacher(aug_img)
        # Compute the normalized teacher output Yˆ given by Yˆc = σc−1(Yc0 − µc) for each c ∈ {1, . . . , 384}
        # normal_t_out = self.compute_normalize_teacher_out(t_out)
        normal_t_out = (t_out-self.channel_mean)/self.channel_std
        s_pdn_out,s_ae_out = student(aug_img)
        # pdb.set_trace()
        # Compute the squared difference between Yˆ and Y A for each tuple (c, w, h) as DAEc,w,h = (Yˆc,w,h − YAc,w,h)2
        distance_ae = torch.pow(normal_t_out-ae_out,2)
        
        # Compute the squared difference between YA and YSTAE for each tuple (c, w, h) as DSTAEc,w,h = (YAc,w,h − Yc,w,hSTAE)2
        distance_stae = torch.pow(ae_out-s_ae_out,2)

        #Compute the loss LAE as the mean of all elements DAE c,w,h of DAE
        LAE = torch.mean(distance_ae)
        #Compute the loss LSTAE as the mean of all elements DSTAE c,w,h of DSTAE
        LSTAE = torch.mean(distance_stae)
        return LAE,LSTAE
    
    def caculate_channel_std(self,dataloader):
        channel_std_ckpt = "{}/good_dataset_channel_std.pth".format(self.ckpt_path)
        if osp.isfile(channel_std_ckpt):
            channel_std = torch.load(channel_std_ckpt)
            self.channel_mean = channel_std['mean'].cuda()
            self.channel_std = channel_std['std'].cuda()
        else:
            self.channel_mean,self.channel_std = self.global_channel_normalize(dataloader)
            print('channel mean:{}'.format(self.channel_mean.shape),'channel std:{}'.format(self.channel_std.shape))
            channel_std = {
                'mean':self.channel_mean,
                'std':self.channel_std
            }
            torch.save(channel_std,channel_std_ckpt)

    def train(self,epochs=100):
        # Initialize Adam [29] with a learning rate of 10−4 and a weight decay of 10−5 for the parameters of S and A

        dataset = MVTecDataset(
                        root=self.mvtech_dir,
                        transform=self.data_transforms,
                        gt_transform=self.gt_transforms,
                        phase='train'
                        )
        dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
        print('load train dataset:length:{}'.format(len(dataset)))
        imagenet = ImageNetDataset(imagenet_dir=self.imagenet_dir,transform=self.data_transforms)
        imagenet_loader = DataLoader(imagenet,batch_size=1,shuffle=True)
        # len_traindata = len(dataset)
        imagenet_iterator = iter(imagenet_loader)
        self.caculate_channel_std(dataloader)
        optimizer = optim.Adam(list(self.student.parameters())+list(self.ae.parameters()),lr=0.0001,weight_decay=0.00001)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.1, last_epoch=int(epochs*0.9))
        best_loss = 100000
        for epoch in range(epochs):

            for i_batch, sample_batched in enumerate(dataloader):
                optimizer.zero_grad()
                image = sample_batched['image'].cuda()
                loss_st = self.loss_st(image,imagenet_iterator,self.teacher,self.student)
                LAE,LSTAE = self.loss_ae(image,self.teacher,self.student,self.ae)
                loss = loss_st + LAE + LSTAE
                loss.backward()
                optimizer.step()
                if i_batch % 10 == 0:
                    print("epoch:{},batch:{},total_loss:{:.4f},loss_st:{:.4f},loss_ae:{:.4f},loss_stae:{:.4f}".format(epoch,i_batch,loss.item(),loss_st.item(),LAE.item(),LSTAE.item()))
            # scheduler.step()
            if epoch % 10 == 0:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    print('saving model in {}'.format(self.ckpt_path))
                    torch.save(self.student.state_dict(),'{}/student.pth'.format(self.ckpt_path))
                    torch.save(self.ae.state_dict(),'{}/autoencoder.pth'.format(self.ckpt_path))
        qa_st,qb_st,qa_ae,qb_ae = self.val()
        quantiles = {
            'qa_st':qa_st,
            'qb_st':qb_st,
            'qa_ae':qa_ae,
            'qb_ae':qb_ae,
            'std':self.channel_std.cpu().numpy(),
            'mean':self.channel_mean.cpu().numpy()
        }
        #save quantiles numpy type
        np.save('{}/quantiles.npy'.format(self.ckpt_path),quantiles)
            
    def val(self):
        xst,xae = [],[]
        dataset = MVTecDataset(
                        root=self.mvtech_dir,
                        transform=self.data_transforms,
                        gt_transform=self.gt_transforms,
                        phase='test'
                        )
        dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
        for i_batch, sample_batched in enumerate(dataloader):
            sample_batched = sample_batched['image'].cuda()
            t_out = self.teacher(sample_batched)
            s_out,_ = self.student(sample_batched)
            ae_out = self.ae(sample_batched)
            # normal_t_out = self.compute_normalize_teacher_out(t_out)
            normal_t_out = (t_out-self.channel_mean)/self.channel_std
            distance_s_t = torch.pow(normal_t_out-s_out,2)
            distance_stae = torch.pow(ae_out-s_out,2)
            # Compute the anomaly maps MST = 384−1 P 384 c=1 Dc ST and MAE = 384−1 P 384 c=1 Dc STAE
            anomaly_map_st_by_c = torch.mean(distance_s_t,dim=1)
            anomaly_map_stae_by_c = torch.mean(distance_stae,dim=1)
            # Resize MST and MAE to 256 × 256 pixels using bilinear interpolation
            anomaly_map_st = F.interpolate(anomaly_map_st_by_c.unsqueeze(0),
                                            size=self.fmap_size,mode='bilinear')
            anomaly_map_ae = F.interpolate(anomaly_map_stae_by_c.unsqueeze(0),
                                            size=self.fmap_size,mode='bilinear')
            # XST ← XST_ vec(MST) . Append to the sequence of local anomaly scores
            xst.append(anomaly_map_st.detach().cpu().numpy())
            # XAE ← XAE_ vec(MAE) . Append to the sequence of local anomaly scores
            xae.append(anomaly_map_ae.detach().cpu().numpy())
        # Compute the 0.9-quantile qa ST and the 0.995-quantile qb ST of the elements of XST.
        # qa_st = torch.quantile(np.concatenate(xst),0.9)
        # qb_st = torch.quantile(np.concatenate(xst),0.995)
        qa_st = np.percentile(np.concatenate(xst),90)
        qb_st = np.percentile(np.concatenate(xst),99.5)
        # Compute the 0.9-quantile qa AE and the 0.995-quantile qb AE of the elements of XAE.
        # qa_ae = torch.quantile(torch.cat(xae),0.9)
        # qb_ae = torch.quantile(torch.cat(xae),0.995)
        qa_ae = np.percentile(np.concatenate(xae),90)
        qb_ae = np.percentile(np.concatenate(xae),99.5)
        return qa_st,qb_st,qa_ae,qb_ae

if __name__ == '__main__':
    ckpt = 'ckpt'
    if not os.path.exists(ckpt):
        os.makedirs(ckpt)
    rst = Reduced_Student_Teacher(
        mvtech_dir="data/MVTec_AD/bottle/",
        imagenet_dir="data/ImageNet/",
        ckpt_path=ckpt,
    )
    rst.train(epochs=200)





        
