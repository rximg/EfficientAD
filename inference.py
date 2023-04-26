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
import numpy as np
from torchvision import transforms
from models import Teacher,Student,AutoEncoder
from data_loader import MVTecDataset
import pdb
import cv2
from PIL import Image
import os
class Inference(object):

    def __init__(self,val_dir,model_path, result_path = 'data/result', model_size='S',resize=256) -> None:
        self.val_dir = val_dir
        self.result_path = result_path
        self.teacher = Teacher(model_size)
        self.student = Student(model_size)
        self.ae = AutoEncoder()
        self.model_path = model_path
        self.load_model()
        self.data_transforms = transforms.Compose([
                        transforms.Resize((resize, resize), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        ])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((resize, resize)),
                        transforms.ToTensor()])

    def load_model(self,):
        teacher_ckpt = torch.load(self.model_path+'/best_teacher.pth')
        student_ckpt = torch.load(self.model_path+'/student.pth')
        ae_ckpt = torch.load(self.model_path+'/autoencoder.pth')
        # pdb.set_trace()
        self.teacher.load_state_dict(teacher_ckpt)
        self.student.load_state_dict(student_ckpt)
        self.ae.load_state_dict(ae_ckpt)
        self.teacher.cuda()
        self.student.cuda()
        self.ae.cuda()
        quantiles = np.load(self.model_path+'/quantiles.npy',allow_pickle=True).item()
        self.qa_st = torch.tensor(quantiles['qa_st']).cuda()
        self.qb_st = torch.tensor(quantiles['qb_st']).cuda()
        self.qa_ae = torch.tensor(quantiles['qa_ae']).cuda()
        self.qb_ae = torch.tensor(quantiles['qb_ae']).cuda()
        self.channel_std = torch.tensor(quantiles['std']).cuda()
        self.channel_mean = torch.tensor(quantiles['mean']).cuda()

    def infer_single(self,sample_batched):
        img = sample_batched['image']
        img = img.cuda()
        teacher_output = self.teacher(img)
        student_output = self.student(img)
        ae_output = self.ae(img)
        #3: Split the student output into Y ST ∈ R 384×64×64 and Y STAE ∈ R 384×64×64 as above
        y_st = student_output[:, :384, :, :]
        y_stae = student_output[:, -384:, :, :]

        normal_teacher_output = (teacher_output-self.channel_mean)/self.channel_std

        distance_st = torch.pow(normal_teacher_output-y_st,2)
        distance_stae = torch.pow(ae_output-y_stae,2)

        fmap_st = torch.mean(distance_st,dim=1,keepdim=True)
        fmap_stae = torch.mean(distance_stae,dim=1,keepdim=True)
        # fmap_st = fmap_st.view(1,1,64,64)
        # fmap_stae = fmap_stae.view(1,1,64,64)
        # pdb.set_trace()
        fmap_st = F.interpolate(fmap_st,size=(256,256),mode='bilinear')
        fmap_stae = F.interpolate(fmap_stae,size=(256,256),mode='bilinear')
        # fmap_st = fmap_st.view(256,256)
        normalized_mst = (0.1*(fmap_st-self.qa_st))/(self.qb_st-self.qa_st)
        normalized_mae = (0.1*(fmap_stae-self.qa_ae))/(self.qb_ae-self.qa_ae)
        combined_map = 0.5*normalized_mst+0.5*normalized_mae
        image_score = torch.max(combined_map)
        return combined_map,image_score

    def infer(self):
        dataset = MVTecDataset(
                        root=self.val_dir,
                        transform=self.data_transforms,
                        gt_transform=self.gt_transforms,
                        phase='test'
                        )
        dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
        num = 0
        for i_batch, sample_batched in enumerate(dataloader):
            combined_map,image_score = self.infer_single(sample_batched)
            print(image_score.item())
            sorted_str = str(int(image_score.item()*10000)).rjust(6,'0')
            if not os.path.exists(self.result_path):
                os.mkdir(self.result_path)
            out_im_path = '{}/{}_{}_{}.png'.format(self.result_path, sorted_str,num,i_batch,image_score)
            
            out_im_np = combined_map[0,0,:,:].cpu().detach().numpy()
            out_im_np = (out_im_np-np.min(out_im_np))/(np.max(out_im_np)-np.min(out_im_np))
            out_im_np = (out_im_np*255).astype(np.uint8)

            origin_img = sample_batched['image'][0].cpu().detach().numpy()
            origin_img = np.transpose(origin_img,(1,2,0))
            origin_img_np = (origin_img*255).astype(np.uint8)
            # origin_img_np = np.array(origin_img)
            color_fmap = cv2.applyColorMap(out_im_np, cv2.COLORMAP_JET)
            out_hstack = np.hstack((origin_img_np,color_fmap))
            cv2.imwrite(out_im_path,out_hstack)


            
        
if __name__ == "__main__":
    val_dir = 'data/MVTec_AD/bottle/'
    model_path = 'ckpt'
    infer = Inference(val_dir,model_path)
    infer.infer()
        
