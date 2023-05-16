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
from torch.utils.data import DataLoader
import numpy as np

import shutil
from torchvision import transforms
from models import Teacher,Student,AutoEncoder
from data_loader import MVTecDataset,get_AD_dataset
import pdb
import cv2
import os
from tqdm import tqdm
import os.path as osp
from sklearn.metrics import roc_auc_score,average_precision_score

class Inference(object):

    def __init__(self,category,val_dir,model_path,ratio=0.1,score_in_mid_size=224,channel=384, result_path = 'data/result', model_size='S',resize=256,device='cuda',dataset_type='MVTec') -> None:
        self.category = category 
        self.ratio = ratio
        self.resize = resize
        self.score_in_mid_size = score_in_mid_size
        self.val_dir = val_dir
        self.channel = channel
        self.dataset_type = dataset_type
        self.result_path = osp.join(result_path,label)
        self.teacher = Teacher(model_size)
        self.student = Student(model_size)
        self.ae = AutoEncoder()
        self.model_path = model_path
        self.device = device
        self.load_model()
        self.data_transforms = transforms.Compose([
                        transforms.Resize((resize, resize)),
                        transforms.ToTensor(),
                        ])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((resize, resize)),
                        transforms.ToTensor()])

    def load_model(self,):
        teacher_ckpt = torch.load(self.model_path+'/best_teacher.pth',map_location=torch.device(self.device))
        student_ckpt = torch.load(self.model_path+'/{}_student.pth'.format(self.category),map_location=torch.device(self.device))
        ae_ckpt = torch.load(self.model_path+'/{}_autoencoder.pth'.format(self.category),map_location=torch.device(self.device))
        self.teacher.load_state_dict(teacher_ckpt)
        self.student.load_state_dict(student_ckpt)
        self.ae.load_state_dict(ae_ckpt)
        self.teacher.eval()
        self.student.eval()
        self.ae.eval()
        self.teacher.to(self.device)
        self.student.to(self.device)
        self.ae.to(self.device)
        quantiles = np.load(self.model_path+'/{}_quantiles.npy'.format(self.category),allow_pickle=True).item()
        self.qa_st = torch.tensor(quantiles['qa_st'],device=self.device)
        self.qb_st = torch.tensor(quantiles['qb_st'],device=self.device)
        self.qa_ae = torch.tensor(quantiles['qa_ae'],device=self.device)
        self.qb_ae = torch.tensor(quantiles['qb_ae'],device=self.device)
        self.channel_std = torch.tensor(quantiles['std'],device=self.device)
        self.channel_mean = torch.tensor(quantiles['mean'],device=self.device)

    def infer_single(self,sample_batched):
        img = sample_batched['image']
        img = img.to(self.device)
        with torch.no_grad():
            teacher_output = self.teacher(img)
            student_output = self.student(img)
            ae_output = self.ae(img)
        #3: Split the student output into Y ST ∈ R 384×64×64 and Y STAE ∈ R 384×64×64 as above
        y_st = student_output[:, :self.channel, :, :]
        y_stae = student_output[:, -self.channel:, :, :]

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
        normalized_mst = (self.ratio*(fmap_st-self.qa_st))/(self.qb_st-self.qa_st)
        normalized_mae = (self.ratio*(fmap_stae-self.qa_ae))/(self.qb_ae-self.qa_ae)
        combined_map = 0.5*normalized_mst+0.5*normalized_mae
        score_start = (self.resize-self.score_in_mid_size)//2
        # pdb.set_trace()
        image_score = torch.max(combined_map[:,:,
            score_start:score_start+self.score_in_mid_size,
            score_start:score_start+self.score_in_mid_size
        ])
        # image_score = torch.max(combined_map)
        return combined_map,image_score

    def eval(self):
        # dataset = MVTecDataset(
        #                 root=self.val_dir,
        #                 transform=self.data_transforms,
        #                 gt_transform=self.gt_transforms,
        #                 phase='test',
        #                 category=self.category
        #                 )
        dataset = get_AD_dataset(
            type=self.dataset_type,
            root=self.val_dir,
            transform=self.data_transforms,
            gt_transform=self.gt_transforms,
            phase='test',
            category=self.category
        )
        dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
        total_pixel_scores = torch.empty(0)
        total_gt_pixel_scores = torch.empty(0)
        num = 0
        scores = []
        gts = []
        print(self.result_path)
        if os.path.exists(self.result_path):
            shutil.rmtree(self.result_path)
        os.makedirs(self.result_path)
        
        for i_batch, sample_batched in tqdm(enumerate(dataloader)):
            gts.append(sample_batched['label'].item())
            name = sample_batched['name'][0]
            label = sample_batched['type'][0]
            total_gt_pixel_scores = torch.cat((total_gt_pixel_scores,sample_batched['gt'].view(-1)))
            combined_map,image_score = self.infer_single(sample_batched)
            scores.append(image_score.item())
            total_pixel_scores = torch.cat((total_pixel_scores,combined_map.detach().cpu().view(-1)))
            out_dir = '{}/{}'.format(self.result_path,label)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_im_path = "{}/{}.png".format(out_dir,name)
            out_im_np = combined_map[0,0,:,:].cpu().detach().numpy()
            out_im_np = ((out_im_np).clip(0,1)*255).astype(np.uint8)
            out_im_np_rgb = cv2.cvtColor(out_im_np,cv2.COLOR_GRAY2RGB)
            out_im_thresh = cv2.threshold(out_im_np, 100, 255, cv2.THRESH_BINARY )[1]
            out_im_thresh = cv2.cvtColor(out_im_thresh,cv2.COLOR_GRAY2RGB)

            origin_img = sample_batched['image'][0].cpu().detach().numpy()
            origin_img = np.transpose(origin_img,(1,2,0))
            origin_img_np = (origin_img*255).astype(np.uint8)
            origin_img_np = cv2.cvtColor(origin_img_np,cv2.COLOR_RGB2BGR)
            color_fmap = cv2.applyColorMap(out_im_np, cv2.COLORMAP_JET)
            origin_with_fmap = cv2.addWeighted(origin_img_np,0.5,color_fmap,0.5,0)
            gt_np = sample_batched['gt'][0].cpu().detach().numpy()
            gt_rgb = cv2.cvtColor((gt_np[0,:,:]*255).astype(np.uint8),cv2.COLOR_GRAY2RGB)
            # pdb.set_trace()
            out_hstack = np.hstack((origin_img_np,gt_rgb,origin_with_fmap,out_im_np_rgb,out_im_thresh))
            cv2.imwrite(out_im_path,out_hstack)
        gtnp = np.array(gts)
        scorenp = np.array(scores)
        total_gt_pixel_scoresnp = total_gt_pixel_scores.cpu().detach().numpy().astype('uint8')
        total_pixel_scoresnp = total_pixel_scores.cpu().detach().numpy()
        # pdb.set_trace()
        auroc = roc_auc_score(gtnp,scorenp)
        if total_gt_pixel_scoresnp.max()==0:
            print("label:{},auroc:{:.4f}".format(self.category,auroc))
            return
        auroc_pixel = roc_auc_score(total_gt_pixel_scoresnp,total_pixel_scoresnp)
        ap_pixel = average_precision_score(total_gt_pixel_scoresnp,total_pixel_scoresnp)
        ap = average_precision_score(gtnp,scorenp)
        print("label:{},auroc:{:.4f},auroc_pixel:{:.4f},ap:{:.4f},ap_pixel:{:.4f}".format(self.category,auroc,auroc_pixel,ap,ap_pixel))        


            
        
if __name__ == "__main__":
    # val_dir = 'data/uniad224data/'
    # model_path = 'ckptS_T'
    # label = "HC_35IL1CROP"
    val_dir = 'data/MVTec_AD/'
    model_path = 'ckptSmall'
    label = "bottle"
    infer = Inference(label,val_dir,model_path,ratio=1,model_size='S',device='cuda')
    infer.eval()
