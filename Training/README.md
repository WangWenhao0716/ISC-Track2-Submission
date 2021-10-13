# The steps for reproducing training

## Generate datasets

In the training parts, we use one dataset. The dataset is selected from the provided training data. We choose 100,000 out of 1000,000 images to perform training. It should be noted that NO reference data is used.

To be convenient, we supply the [**link**]() to the generated dataset as follows. You can directly download them from Google drive and unzip them. The default path is ```/dev/shm``` to store the images temporarily for training.

Or you can generate the training datasets according to the codes in the ```generate``` folder by yourself. It takes about one day to generate it using one core of Intel 
