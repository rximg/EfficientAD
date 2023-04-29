from train_reduced_student import Reduced_Student_Teacher
import os
categorys = [
    'bottle',
    'cable',
    'capsule',
    'carpet',
    'grid',
    'hazelnut',
    'leather',
    'metal_nut',
    'pill',
    'screw',
    'tile',
    'toothbrush',
    'transistor',
    'wood',
    'zipper'
]
if __name__ == '__main__':
    ckpt = 'ckpt'
    if not os.path.exists(ckpt):
        os.makedirs(ckpt)

    for category in categorys:
        rst = Reduced_Student_Teacher(
            category,
            mvtech_dir="data/MVTec_AD/",
            imagenet_dir="data/ImageNet/",
            ckpt_path=ckpt,
        
            batch_size=8,
        )
        rst.train()