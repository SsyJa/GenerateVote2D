# GenerateVote

We compute instance centroids from SAM-generated object masks in the image.



## 0. Environment

```yml
name: vote_generate_2d
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

