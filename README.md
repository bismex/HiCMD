# Hi-CMD: Hierarchical Cross-Modality Disentanglement for Visible-Infrared Person Re-Identification

The code is based on the PyTorch implementation of the [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) and [Cross-Model-Re-ID-baseline](https://github.com/mangye16/Cross-Modal-Re-ID-baseline).


## Prerequisites
- Ubuntu 18.04
- Python 3
- Pytorch (recent version is recommended)
- NVIDIA GPU (>8.5GB)
- CUDA 10.0
- CUDNN 7.5


## Getting Started
### Installation

- Configure virtual (anaconda) environment

```bash
conda create -n env_name python=3.6
source activate env_name
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

- Install python libraries

```bash
conda install -c conda-forge matplotlib
conda install -c anaconda yaml
conda install -c anaconda pyyaml 
conda install -c anaconda scipy
conda install -c anaconda pandas 
conda install -c anaconda scikit-learn 
conda install -c conda-forge opencv
conda install -c anaconda seaborn
conda install -c conda-forge tqdm
```

- Install ["Pretrained models for Pytorch"](https://github.com/Cadene/pretrained-models.pytorch)
  - To use pretrained models
```bash
git clone https://github.com/Cadene/pretrained-models.pytorch.git
cd pretrained-models.pytorch
python setup.py install


- Clone this repo:
```bash
git clone https://github.com/bismex/ongoing_project.git
```
### Dataset
#### Download the datasets

- [DB](https://drive.google.com/open?id=1sEe5DQC5rJNHYuNKLOgkAj2iEg7aFMTy)
- RegDB Dataset [1] : The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.
  - (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website).
- SYSU-MM01 Dataset [2] : The SYSU-MM01 dataset can be downloaded from this [website](http://www.sysu.edu.cn/403.html).

#### Prepare the datasets
```bash
python prepare.py 
```

## Training

## Testing

## Reference

- [1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

- [2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.
