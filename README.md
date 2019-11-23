# Hi-CMD: Hierarchical Cross-Modality Disentanglement for Visible-Infrared Person Re-Identification


## Prerequisites
- Ubuntu 18.04
- Python 3.6
- PyTorch 1.0+ (recent version is recommended)
- NVIDIA GPU (>= 8.5GB)
- CUDA 10.0 (optional)
- CUDNN 7.5 (optional)


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
```

- Clone this repo:

---

## Training and testing

- In the actual experiment, a total of 10 experiment sets are used.
- Here is a simple example of running an experiment on only one set (RegDB-01).
- Download [RegDB_01](https://drive.google.com/open?id=1sEe5DQC5rJNHYuNKLOgkAj2iEg7aFMTy) (for a simple test) 
  - The RegDB_01 dataset should be included in './data/'
  - Ex: `./HiCMD/dataset/RegDB`

### Training

```bash
sh train.sh
```

### Testing on pretrained model

#### RegDB_01

- The pretrained RegDB_01 model should be included.
- Download []()
  - The pretrained RegDB_01 model should be included in './pretrained/'
  - Ex: `./HiCMD/pretrained/checkpoints`

```bash
sh test.sh
```

- The code provides the following results.
  - Rank1: 70.44\%
  - Rank5: 79.37\%
  - Rank10: 85.15\%
  - Rank20: 91.55\%
  - mAP: 65.93\%
- The performance of the manuscript (Rank1: 70.93\%) is obtained by averaging this experiment for 10 sets.
- If the code is not working, please refer to './pretrained/test_results/net_70000_RegDB_01_(ms1.0)_f1_test_result.txt'

#### SYSU-MM01

- MATLAB is required for evaluating SYSU-MM01 (official code).
- Download []()
  - The pretrained SYSU-MM01 model should be included in './eval_SYSU/'
  - Ex: `./HiCMD/eval_SYSU/`


- The code provides the following results.
  - Rank1: 34.94\%
  - Rank5: 65.48\%
  - Rank10: 77.58\%
  - Rank20: 88.38\%
  - mAP: 35.94\%
  
- If the code is not working, please refer to './eval_SYSU/results_test_SYSU.txt'

---



## (Optional)

- If you want to experiment with all sets of RegDB, download the official dataset:
  - The RegDB dataset [1] can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form. (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website).

- If you want to experiment with SYSU-MM01, download the official dataset:
  - The SYSU-MM01 dataset [2] can be downloaded from this [website](http://www.sysu.edu.cn/403.html).
  - The authors' official [matlab code](https://github.com/wuancong/SYSU-MM01) is used to evaluate the SYSU dataset.

- Change the 'data_name' from 'RegDB_01' to the name of other datasets.
- Process the downloaded data according to the code by `python prepare.py`.
- Train and test 


## Acknowledgement

The code is based on the PyTorch implementation of the [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch), [Cross-Model-Re-ID-baseline](https://github.com/mangye16/Cross-Modal-Re-ID-baseline), [MUNIT](https://github.com/NVlabs/MUNIT), [DGNET](https://github.com/NVlabs/DG-Net), [SYSU-evaluation](https://github.com/wuancong/SYSU-MM01).


## Reference


- [1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

- [2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.
