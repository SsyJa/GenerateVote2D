# VoteSplat

*<!-- We compute instance centroids from SAM-generated object masks in the image. -->*

We propose VoteSplat, a novel approach for scene understanding in 3DGS representations by incorporating effective and distinctive voting mechanisms into 3DGS. 

## 0. Overview

![image-20250306190307133](https://gitee.com/syjia_xdu/picgo-imgs/raw/master/imgs/202503061903313.png)

## 1. Environment

```yml
name: generate_vote_2d

channels:
  - conda-forge
  - defaults
dependencies:

  - numpy
  - opencv
  - pathlib
  - tqdm
  - matplotlib
```

