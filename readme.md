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

## Disclaimer
This repository is implemented according to the pseudocode in the appendix, which may differ from the official version and has not been tested for performance. It does not represent the true effectiveness of EfficientAD. If you have any suggestions, please submit an issue or PR.


