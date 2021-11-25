# ISC-Track2-Submission (Rank 3)
The codes and related files to reproduce the results for Image Similarity Challenge Track 2.

2021.11.25 Updates: This solution is verified! If you find this code useful for your research, please cite our paper.
2021.11.24 Updates: Fix some bugs without changing performance.

## Required dependencies
To begin with, you should install the following packages with the specified versions in Python, Anaconda. Please do not use cuda 11.0, which has some bugs. The hardware chosen is Nvidia Tesla V100 and Intel CPU. We also reproduce the experiments using DGX A100 with AMD CPU, with pytorch 1.9.1 and cuda 11.1.

* python 3.7.10
* pytorch 1.7.1 with cuda 10.1
* faiss-gpu 1.7.1 with cuda 10.1
* h5py 3.4.0
* pandas 1.3.3
* sklearn 1.0
* skimage 0.18.3
* PIL 8.3.2
* cv2 4.5.3.56
* numpy 1.16.0
* torchvision 0.8.2 with cuda 10.1
* augly 0.1.4
* selectivesearch 0.4
* face-recognition 1.3.0 (with dlib of gpu-version)
* tqdm 4.62.3
* requests 2.26.0
* seaborn 0.11.2
* mkl 2.4.0
* loguru 0.5.3

Note: Some unimportant packages may be missing, please install them using pip directly when an error occurs.

## Pre-trained models
The pre-trained models we used is directly downloaded from [**here**](https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/resnet50.pth). It is supplied by Facebook Research, and the project is [**Barlow Twins**](https://github.com/facebookresearch/barlowtwins). You should rename it to ```resnet50_bar.pth```.

## Training
For training, we generate one dataset. The training process takes less than one day on 4 V100 GPUs. The whole training codes, including how to generate training dataset and the link to the generated dataset, are given in the ```Training``` folder. For more details, please refer to the readme file in that folder.

## Test
To test the performance of the trained model, we perform multi-scale testing and ensemble all the features to get the final representation. We give all the information to generate our final results in the ```Test``` folder. Please reproduce the results according to the readme file in that folder.

## Citation
```
@article{wang2021bag,
  title={Bag of Tricks and A Strong baseline for Image Copy Detection},
  author={Wang, Wenhao and Zhang, Weipu and Sun, Yifan and Yang, Yi},
  journal={arXiv preprint arXiv:2111.08004},
  year={2021}
}
```
