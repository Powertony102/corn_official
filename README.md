# Leveraging CORAL-Correlation Consistency Network for Semi-Supervised Left Atrium MRI Segmentation

[Paper(Arxiv)](https://arxiv.org/abs/2410.15916) [Paper(IEEE)](https://ieeexplore.ieee.org/abstract/document/10822694)

## Introduction

This repository is for our paper: *Leveraging CORAL-Correlation Consistency Network for Semi-Supervised Left Atrium MRI Segmentation*

As for the first author, it is his first published paper, which finished in his year 1 of university life.



## Requirements

This repository is based on:

- PyTorch 2.1.0
- CUDA 12.1
- Python 3.11.6

All experiments in our paper were conducted on a single NVIDIA A100 80GB PCIe GPU



## Usage

### Training

1. Clone the repo

   ```
   https://github.com/Powertony102/corf_official.git
   ```

2. Put the data in './CORN/data'; (You may need to create it.)

3. Train the model

   ``` 
   cd CORN
   # e.g., for 5% labels on LA
   python code/train_corn_3d.py --labelnum 4 --gpu 0 --seed 1337
   ```

### Testing

```
cd CORN
# e.g., for 5% labels on LA
python code/test_3d.py --labelnum 4 --gpu 0 --seed 1337
```



## Acknowledgements:

Our source code is origin from: [UAMT](https://github.com/yulequan/UA-MT), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS),[MC-Net](https://https//github.com/ycwu1997/MC-Net) and [CAML](https://github.com/Herschel555/CAML/tree/master?tab=readme-ov-file). Thanks for these authors for their valuable works.
