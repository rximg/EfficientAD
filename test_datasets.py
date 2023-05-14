from data_loader import load_infinite,get_AD_dataset
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms   
import pdb
resize = 256     
data_transforms = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                ])
gt_transforms = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor()])
dataset = get_AD_dataset(
                    type='MVTecLoco',
                    root='data/mvtec_loco_anomaly_detection',
                    transform=data_transforms,
                    gt_transform=gt_transforms,
                    phase='test',
                    category='breakfast_box'
                    )

train_dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0)
dataiter = load_infinite(train_dataloader)
data = next(dataiter)
pdb.set_trace()
# print(data)