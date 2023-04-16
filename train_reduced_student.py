import torch
import torch.nn.functional as F
import random
from torch import optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from models import Teacher,Student,AutoEncoder
from data_loader import TrainImageOnlyDataset,ImageNetDataset


def channel_normalization_parameters(teacher,dataloader):

# for c ∈ 1, . . . , 384 do . Compute teacher channel normalization parameters µ ∈ R
# 384 and σ ∈ R
# 384
# 4: Initialize an empty sequence X ← ( )
# 5: for Itrain ∈ Itrain do
# 6: Y
# 0 ← T(Itrain)
# 7: X ← X_ vec(Yc
# 0
# ) . Append the channel output to X
# 8: end for
# 9: Set µc to the mean and σc to the standard deviation of the elements of X
    vects = []
    for i_batch, sample_batched in enumerate(dataloader):
        vect = teacher(sample_batched)
        b,c,h,w = vect.shape
        vect = vect.view(b,c,-1)
        vects.append(vect)
    vects = torch.cat(vects,dim=0)
    channel_mean = torch.mean(vects,dim=0)
    channel_std = torch.std(vects,dim=0)
    return channel_mean,channel_std


class Reduced_Student_Teacher(object):


    def __init__(self,train_dir,val_dir,imagenet_dir,ckpt_path,channel_size=384,fmap_size  = 256) -> None:
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.imagenet_dir = imagenet_dir
        self.teacher = Teacher()
        self.load_pretrain_teacher()
        self.student = Student()
        self.ae = AutoEncoder()
        self.ckpt_path = ckpt_path
        self.channel_size = channel_size
        self.fmap_size = (fmap_size,fmap_size)
        self.channel_mean,self.channel_std = None,None

    def load_pretrain_teacher(self):
        self.teacher.load_state_dict(torch.load(self.ckpt_path+'/teacher.pth'))
        self.teacher.eval()


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
        for c in range(self.channel_size):
            c_out = t_out[:,c,:,:]
            normal_t_out.append((c_out-self.channel_mean[c])/self.channel_std[c])
        normal_t_out = torch.stack(normal_t_out,dim=1)
        return normal_t_out

    def loss_st(self,image,imagenet_loader,teacher,student,autoencoder):
        #Choose a random training image_train from dataloader_rain
        # image = self.dataloader[0]
        #Forward pass of the student–teacher pair
        t_pdn_out = teacher(image)
        b,c,h,w = t_pdn_out.shape
        #Compute the normalized teacher output given by normal_t_out = (teacher_out_channel − mean_c)/std_c for each channcel ∈ {1, . . . , 384}
        
        normal_t_out = self.compute_normalize_teacher_out(t_pdn_out)
        s_pdn_out,s_ae_out = student(image)
        # Compute the squared difference between normal_t_out and s_pdn_out for each tuple (c, w, h) as DST c,w,h = (Yˆc,w,h − YSTc,w,h)2
        distance_s_t = torch.pow(normal_t_out-s_pdn_out,2)
        # Compute the 0.999-quantile of the elements of DST, denoted by dhard
        dhard = torch.quantile(distance_s_t,0.999)
        # Compute the loss Lhard as the mean of all DST c,w,h ≥ dhard
        Lhard = torch.mean(distance_s_t[distance_s_t>=dhard])
        # Choose a random pretraining image P ∈ R 3×256×256 from ImageNet [54]
        image_p = next(imagenet_loader)
        s_imagenet_out = student(image_p)
        # Compute the loss LST = Lhard + (384 · 64 · 64)−1 P 384 c=1 k S(P)ck 2 F
        loss_st = Lhard + (1/(c*h*w))*torch.sum(torch.pow(s_imagenet_out,2))
        return loss_st
    
    def loss_ae(self,image,teacher,student,autoencoder):
        aug_img = self.choose_random_aug_image(image=image)
        ae_out = autoencoder(aug_img)
        t_out = teacher(aug_img)
        # Compute the normalized teacher output Yˆ given by Yˆc = σc−1(Yc0 − µc) for each c ∈ {1, . . . , 384}
        normal_t_out = self.compute_normalize_teacher_out(t_out)
        s_pdn_out,s_ae_out = student(aug_img)
        
        # Compute the squared difference between Yˆ and Y A for each tuple (c, w, h) as DAEc,w,h = (Yˆc,w,h − YAc,w,h)2
        distance_ae = torch.pow(normal_t_out-ae_out,2)
        
        # Compute the squared difference between YA and YSTAE for each tuple (c, w, h) as DSTAEc,w,h = (YAc,w,h − Yc,w,hSTAE)2
        distance_stae = torch.pow(ae_out-s_ae_out,2)

        #Compute the loss LAE as the mean of all elements DAE c,w,h of DAE
        LAE = torch.mean(distance_ae)
        #Compute the loss LSTAE as the mean of all elements DSTAE c,w,h of DSTAE
        LSTAE = torch.mean(distance_stae)
        return LAE,LSTAE
    
    def train(self,epochs=100):
        # Initialize Adam [29] with a learning rate of 10−4 and a weight decay of 10−5 for the parameters of S and A
        optimizer = optim.Adam(list(self.student.parameters())+list(self.ae.parameters()),lr=0.0001,weight_decay=0.00001)
        dataset = TrainImageOnlyDataset(root_dir=self.root_dir,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),  
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                            ]))
        dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
        imagenet = ImageNetDataset(root_dir=self.root_dir,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),  
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                            ]))
        imagenet_loader = DataLoader(imagenet,batch_size=1,shuffle=True)
        self.channel_mean,self.channel_std = channel_normalization_parameters(self.teacher,dataloader)

        for epoch in range(epochs):

            for i_batch, sample_batched in enumerate(dataloader):
                optimizer.zero_grad()
                loss_st = self.loss_st(sample_batched,
                                        self.teacher,self.student,self.ae)
                LAE,LSTAE = self.loss_ae(sample_batched,imagenet_loader,
                                         self.teacher,self.student,self.ae)
                loss = loss_st + LAE + LSTAE
                loss.backward()
                optimizer.step()
                if i_batch % 10 == 0:
                    print(i_batch,loss.item())
                    #save model
            if epoch % 10 == 0:
                print('saving model in {}'.format(self.save_dir))
                torch.save(self.student.state_dict(),'student.pt')
                torch.save(self.ae.state_dict(),'ae.pt')
            qa_st,qb_st,qa_ae,qb_ae = self.val()
        quantiles = {
            'qa_st':qa_st,
            'qb_st':qb_st,
            'qa_ae':qa_ae,
            'qb_ae':qb_ae,
            'std':self.channel_std,
            'mean':self.channel_mean
        }
        #save quantiles pytorchtype
        torch.save(quantiles,'quantiles.pt')
            
    def val(self):
        xst,xae = [],[]
        dataset = TrainImageOnlyDataset(
            root_dir=self.val_dir
        )
        dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
        for i_batch, sample_batched in enumerate(dataloader):
            t_out = self.teacher(sample_batched)
            s_out,_ = self.student(sample_batched)
            ae_out = self.ae(sample_batched)
            normal_t_out = self.compute_normalize_teacher_out(t_out)
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
            xst.append(anomaly_map_st)
            # XAE ← XAE_ vec(MAE) . Append to the sequence of local anomaly scores
            xae.append(anomaly_map_ae)
        # Compute the 0.9-quantile qa ST and the 0.995-quantile qb ST of the elements of XST.
        qa_st = torch.quantile(torch.cat(xst),0.9)
        qb_st = torch.quantile(torch.cat(xst),0.995)
        # Compute the 0.9-quantile qa AE and the 0.995-quantile qb AE of the elements of XAE.
        qa_ae = torch.quantile(torch.cat(xae),0.9)
        qb_ae = torch.quantile(torch.cat(xae),0.995)
        return qa_st,qb_st,qa_ae,qb_ae







        
