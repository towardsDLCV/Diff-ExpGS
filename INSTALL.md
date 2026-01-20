# 🛠️ Installation

### 💻 Dependencies Installation

This repository is implemented in **PyTorch 2.7.0** and has been tested on **Ubuntu 22.04** with **Python 3.10**, **CUDA 12.8**, and an **RTX 5090** GPU.

#### 1. Clone the repository

```bash
git clone https://github.com/towardsDLCV/Diff-ExpGS.git --recursive
cd Diff-ExpGS
```

#### 2. Create conda environment

The Conda environment can be recreated using the provided `environment.yaml` file.

```bash
conda env create -f environment.yaml
conda activate ExpGS
```

#### 3. Download Checkpoints 

Download the checkpoints for the [**QuadPrior**](https://github.com/daooshee/QuadPrior/tree/main) diffusion model.

Place the downloaded `.ckpt`, `.yaml`, and `.pkl` files into the `resources` directory to match the structure below:

```text
resources
├── checkpoints
│   ├── COCO-final.ckpt
│   └── main-epoch=00-step=7000.ckpt
│
├── models
│   ├── control_sd15_ini-001.ckpt
│   └── cldm_v15.yaml
│
└── empty_embedding.pkl
```

### 🗂️ Dataset Download and Preparation

Diff-ExpGS is evaluated on the **LOM** and **NeRF360** benchmark under three exposure conditions: **low exposure**, **over-exposure**, and **varying exposure**.

- [**Low and Over-Exposure LOM Dataset**](https://drive.google.com/file/d/19egA_g1LFP0JAEX5_fV5YQEG5GJnNpX3/view?usp=drive_link)
- [**Varying-Exposure NeRF360 Dataset**](https://drive.google.com/file/d/1941tWJi2rKMGt34w5l6OfQlom5eYt-Yc/view?usp=drive_link)

After downloading, please organize the datasets as follows:

```text
data
├── LOM_full
│   ├── bike
│   │   ├── high
│   │   ├── low
│   │   ├── over_exp
│   │   └── ...
│   ├── buu
│   ├── chair
│   ├── shrub
│   └── sofa
│
├── NeRF_360
│   ├── bicycle
│   │   ├── images
│   │   ├── images_8
│   │   ├── sparse
│   │   ├── poses_bounds.npy
│   │   └── ...
│   ├── bonsai
│   ├── counter
│   ├── garden
│   ├── kitchen
│   ├── room
└   └── stump
```

### 🔗 Dataset Citation

Please cite the following works when using these datasets, as our data preparation follows Aleth-NeRF and Luminance-GS:

```tex
@inproceedings{cui_aleth_nerf,
  title     = {Aleth-NeRF: Illumination Adaptive NeRF with Concealing Field Assumption},
  author    = {Cui, Ziteng and Gu, Lin and Sun, Xiao and Ma, Xianzheng and Qiao, Yu and Harada, Tatsuya},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2024}
}

@inproceedings{cui_luminance_gs,
  title     = {Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment},
  author    = {Cui, Ziteng and Chu, Xuangeng and Harada, Tatsuya},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```
