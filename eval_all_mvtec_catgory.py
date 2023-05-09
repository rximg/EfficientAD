from eval import Inference
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
    val_dir = 'data/MVTec_AD/'
    model_path = 'ckptSmall'
    for category in categorys:
        try:
            infer = Inference(category,val_dir,model_path,ratio=0.1,model_size='M')
            infer.eval()
        except Exception as e:
            pass
