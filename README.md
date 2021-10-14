# ISC-Track2-Submission
The codes and related files to reproduce the results for Image Similarity Challenge Track 2.

## Required dependencies
To begin with, you should install the following packages with the specified versions in Python, Anaconda. Other versions may work but please do NOT try. For instance, cuda 11.0 has some bugs which bring very bad results. The hardware chosen is Nvidia Tesla V100. Other hardware, such as A100, may work but please do NOT try. The stability is not guaranteed, for instance, the Ampere architecture is not suitable and some instability is observed.

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
* face-recognition 1.3.0
* tqdm 4.62.3
* pyyaml 5.4.1

Note: Some unimportant packages may be missing, please install them using pip directly when an error occurs.

## Pre-trained models
The pre-trained models we used is directly downloaded from [**here**](https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/resnet50.pth). It is supplied by Facebook Research, and the project is [**Barlow Twins**](https://github.com/facebookresearch/barlowtwins).

## Training
For training, we generate one dataset. The training process takes less than one day on 4 V100 GPUs. The whole training codes, including how to generate training dataset and the link to the generated dataset, are given in the ```Training``` folder. For more details, please refer to the readme file in that folder.

## Test
To test the performance of the trained model, we perform multi-scale testing and ensemble all the features to get the final representation. We give all the information to generate our final results in the ```Test``` folder. Please reproduce the results according to the readme file in that folder.


