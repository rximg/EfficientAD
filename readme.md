This project is an unofficial implementation of ["EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies"](https://paperswithcode.com/paper/efficientad-accurate-visual-anomaly-detection),  which is implemented step-by-step according to the pseudocode in the appendix

## Datasets
./data 
- ImageNet
    - n01440764
    - n01443537
    ... 

- MVTec_AD
    - bottle
        - ground_truth
        - test
        - train
    - cable
        - ground_truth
        - test
        - train
    ... 
- result

## Quick start

#### 1. Install PyTorch environment
```
conda activate <your_env>
pip install -r requirements.txt
```

#### 1. Distill a PDN architecture teacher network from wide_resnet101
```
python distillaion_training.py
```

#### 2. train the student network and autoencoder network
```
python train_reduced_student.py
```

## Some results
MVTec bottle
```
image_AUROC: 1.0
image_F1Score: 1.0
pixel_AUROC: 0.9876494407653809
pixel_F1Score: 0.7927650213241577
```
![](https://user-images.githubusercontent.com/54716527/235113149-1c33a160-4da0-4a48-8586-0e34e033fc63.png)
![](https://user-images.githubusercontent.com/54716527/235113227-a88648f9-804a-4b53-aef5-169846661526.png)
![](https://user-images.githubusercontent.com/54716527/235113302-2ef6b2c3-4abd-4e3a-9ce4-f6accead5f26.png)
