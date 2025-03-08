# VoteSplat: Hough Voting Gaussian Splatting for 3D Scene Understanding

![image](https://github.com/user-attachments/assets/84d57f98-1c72-4366-a485-d767a9fc138d)

*<!-- We compute instance centroids from SAM-generated object masks in the image. -->*

We propose VoteSplat, a novel approach for scene understanding in 3DGS representations by incorporating effective and distinctive voting mechanisms into 3DGS. 


## 0. Environment

```yml
name: VoteSplat

channels:
  - conda-forge
  - defaults
dependencies:

  - numpy
  - opencv
  - pathlib
  - tqdm
  - matplotlib
  - cudatoolkit=11.6
  - plyfile
  - python=3.7.13
  - pip=22.3.1
  - pytorch=1.12.1
  - torchaudio=0.12.1
  - torchvision=0.13.1
  - tqdm
  - pip:
    - submodules/diff-gaussian-rasterization
    - submodules/simple-knn
    - submodules/fused-ssim

```



## 1. Running

To generate 2d votes, you could use

```cmd
python preprocess.py
```

To run the optimizer, simply use

```cmd
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```



## 2. Preprocess

Before calculating 2D Votes, we used SAM to extract the masks of different instances in the image based on the pre-processing process in LangSplat. 
