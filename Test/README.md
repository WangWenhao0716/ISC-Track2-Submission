# The steps for reproducing testing

## Trained models
We first provide the trained model to facilitate reproducing. The trained model is obtained according to the provided training codes in training parts by ourselves. You can download it from [**here**](https://drive.google.com/file/d/1FGOfqOckHWVUtvQEgAkDj7TxPTINDIKb/view?usp=sharing).


## Generate dataset
We augment query datasets to deal with overlay images. The codes for augmentation are given in ```yolov5``` folder. For more details, please refer to the readme in that folder. By the way, to run theses augmentations, it is assumed that all the original query images and generated images are saved in ```/dev/shm```.

## Test
