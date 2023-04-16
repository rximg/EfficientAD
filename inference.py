# Require: T, S, A, µ, σ, qa ST, qb ST, qa AE, and qb AE, as returned by Algorithm 
# 1 Require: Test image Itest ∈ R 3×256×256 1: Y 0 ← T(Itest), Y S ← S(Itest), Y A ← A(Itest) 
# 2: Compute the normalized teacher output Yˆ given by Yˆ c = (Yc 0 − µc)σc −1 for each c ∈ {1, . . . , 384}
# 3: Split the student output into Y ST ∈ R 384×64×64 and YSTAE ∈ R 384×64×64 as above 
# 4: Compute the squared difference DST c,w,h = (Yˆ c,w,h − Y ST c,w,h) 2 for each tuple (c, w, h)
# 5: Compute the squared difference DSTAE c,w,h = (Y A c,w,h − Yc,w,h STAE) 2 for each tuple (c, w, h)
# 6: Compute the anomaly maps MST = 384−1 P 384 c=1 Dc ST and MAE = 384−1 P 384 c=1 Dc STAE
# 7: Resize MST and MAE to 256 × 256 pixels using bilinear interpolation
# 8: Compute the normalized ˆMST = 0.1(MST − qa ST)(qb ST − qa ST) −1
# 9: Compute the normalized ˆMAE = 0.1(MAE − qa AE)(qb AE − qa AE) −1
# 10: Compute the combined anomaly map M = 0.5 ˆMST + 0.5 ˆMAE
# 11: Compute the image-level score as mimage = maxi,j Mi,j
# 12: return M and mimage
import torch
import torch.nn.functional as F
import random
from torch import optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from models import Teacher,Student,AutoEncoder
from data_loader import TrainImageOnlyDataset

class Inference(object):

    def __init__(self,val_dir,model_path) -> None:
        self.val_dir = val_dir
        self.teacher = Teacher()
        self.student = Student()
        self.ae = AutoEncoder()
        self.model_path = model_path
        self.load_model()

    def load_model(self,):
        teacher_ckpt = torch.load(self.model_path+'/teacher.pth')
        student_ckpt = torch.load(self.model_path+'/student.pth')
        ae_ckpt = torch.load(self.model_path+'/ae.pth')
        self.teacher.load_state_dict(teacher_ckpt['model_state_dict'])
        self.student.load_state_dict(student_ckpt['model_state_dict'])
        self.ae.load_state_dict(ae_ckpt['model_state_dict'])
        # quantiles = {
        #     'qa_st':qa_st,
        #     'qb_st':qb_st,
        #     'qa_ae':qa_ae,
        #     'qb_ae':qb_ae,
        #     'std':self.channel_std,
        #     'mean':self.channel_mean
        # }
        quantiles = torch.load(self.model_path+'/quantiles.pth')
        self.qa_st = quantiles['qa_st']
        self.qb_st = quantiles['qb_st']
        self.qa_ae = quantiles['qa_ae']
        self.qb_ae = quantiles['qb_ae']
        self.channel_std = quantiles['std']
        self.channel_mean = quantiles['mean']

    def compute_normalize_teacher_out(self,t_out):
        normal_t_out = []
        for c in range(self.channel_size):
            c_out = t_out[:,c,:,:]
            normal_t_out.append((c_out-self.channel_mean[c])/self.channel_std[c])
        normal_t_out = torch.stack(normal_t_out,dim=1)
        return normal_t_out

    def infer_single(self,sample_batched):
        img = sample_batched['image']
        img = img.cuda()
        teacher_output = self.teacher(img)
        student_output,stae_output = self.student(img)
        ae_output = self.ae(img)
        normal_teacher_output = self.compute_normalize_teacher_out(teacher_output)
        distance_st = torch.pow(student_output-ae_output,2)
        distance_stae = torch.pow(normal_teacher_output-stae_output,2)
        fmap_st = torch.mean(distance_st,dim=1)
        fmap_stae = torch.mean(distance_stae,dim=1)
        # fmap_st = fmap_st.view(1,1,64,64)
        # fmap_stae = fmap_stae.view(1,1,64,64)
        fmap_st = F.interpolate(fmap_st,size=(256,256),mode='bilinear')
        fmap_stae = F.interpolate(fmap_stae,size=(256,256),mode='bilinear')
        # fmap_st = fmap_st.view(256,256)
        normalized_mst = 0.1*(fmap_st-self.qa_st)*(self.qb_st-self.qa_st)
        normalized_mae = 0.1*(fmap_stae-self.qa_ae)*(self.qb_ae-self.qa_ae)
        combined_map = 0.5*normalized_mst+0.5*normalized_mae
        image_score = torch.max(combined_map)
        return combined_map,image_score

    def infer(self):
        dataset = TrainImageOnlyDataset(self.val_dir)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
        for i_batch, sample_batched in enumerate(dataloader):
            combined_map,image_score = self.infer_single(sample_batched)
            print(image_score)
            
        
        
