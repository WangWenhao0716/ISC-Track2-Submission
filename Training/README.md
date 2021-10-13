# The steps for reproducing training

## Generate datasets

In the training part, we use one dataset. The dataset is selected from the provided training data. We choose 100,000 out of 1000,000 images to perform training. It should be noted that NO reference data is used.

To be convenient, we supply the [**link**](https://drive.google.com/file/d/1Ianqo1TS3Idx-211oWNANoZJ94WVgx6N/view?usp=sharing) to the generated dataset. You can directly download them from Google drive and unzip them. The default path is ```/dev/shm``` to store the images temporarily for training.

Or you can generate the training datasets according to the codes in the ```generate``` folder by yourself. It takes about one day to generate it using one core of Intel Golden 6240 CPU. To speed up, using multi-cores is a feasible way. We use some images from [**OpenImage**](https://opensource.google/projects/open-images-dataset) to generate overlay and underlay augmentation under CC-by 4.0 License. It should be noted that the using of OpenImage is not a must, other images show similar performance. The part of OpenImage we used can be downloaded from [**here**](https://drive.google.com/file/d/102JynPEzqiZ83zAdquFbrQah2JbXFOuu/view?usp=sharing). 

Assuming all the datasets are stored in ```/dev/shm```, to generate the dataset used for training, you can
```
cd generate && python isc_100k_256.py
```

## Training

Remember that we have one pre-trained model, i.e. ```resnet50_bar.pth```, stored in ```/dev/shm```. You can direct train by
```
bash Train.sh
```
Take a look to the ```Train.sh```
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_high_balance_matrix_distill.py \
-ds isc_100k_256 -a resnet50 --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 5 \
--lr 0.00035 --iters 8000 --epochs 25 \
--data-dir /dev/shm/ \
--logs-dir logs/baseline/50 \
--height 256 --width 256
```

The ```/dev/shm``` is the dir to store images, for ```isc_100k_256``` dataset, please check the number of images is 2,000,000. The checkpoints will be saved into ```logs/baseline/50```. And the final checkpoint, i.e. ```checkpoint_24.pth.tar```, will be used to test. Please do NOT change any hyper-parameters in the script. We support resume training from a checkpoint, and the process can be finished automatically by adding ```--auto_resume```.

Also, to be efficient, you should use the ```Tran.py``` to discard all the fully-connected layers. You should change the path to ```checkpoint_24.pth.tar``` and the path to save by yourself, the following is an example:

```
import torch
mod = torch.load('logs/baseline/50/checkpoint_24.pth.tar',map_location='cpu')
mod['state_dict'].pop('classifier_0.weight')
mod['state_dict'].pop('classifier_1.weight')
mod['state_dict'].pop('classifier_2.weight')
mod['state_dict'].pop('classifier_3.weight')
torch.save(mod['state_dict'], '/dev/shm/baseline_distill_50.pth.tar')
```

I promise the training experiment has been reproduced by ourselves and the results are stable. If you find any problems with the reproduction of training, please feel free to contact me.


