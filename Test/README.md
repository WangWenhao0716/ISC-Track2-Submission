# The steps for reproducing testing

## Trained models
We first provide the trained model to facilitate reproducing. The trained model is obtained according to the provided training codes in training parts by ourselves. You can download it from [**here**](https://drive.google.com/file/d/1FGOfqOckHWVUtvQEgAkDj7TxPTINDIKb/view?usp=sharing).


## Generate dataset
We augment query datasets to deal with overlay images. The codes for augmentation are given in ```yolov5``` folder. For more details, please refer to the readme in that folder. By the way, to run theses augmentations, it is assumed that all the original query images and generated images are saved in ```/dev/shm```.

## Test
Until now, we have one trained model saved in ```Test/baseline_matrix_baro_c24.pth.tar```, and one augmented dataset saved in ```/dev/shm/query_images_detection```. We perform extracting features using 4 different sizes, i.e. 200, 256, 320, 400. Also, we use the features of training images to perform descriptor stretching. The steps to get the final 256-descriptors are as follows.

### Extract features of training images
```
bash extract_training_matrix_400.sh
bash extract_training_matrix_320.sh
bash extract_training_matrix_256.sh
bash extract_training_matrix_200.sh
```

### Extract features of reference images
```
bash extract_reference_matrix_400.sh
bash extract_reference_matrix_320.sh
bash extract_reference_matrix_256.sh
bash extract_reference_matrix_200.sh
```

### Extract features of query images
```
bash extract_query_matrix_400.sh
bash extract_query_matrix_320.sh
bash extract_query_matrix_256.sh
bash extract_query_matrix_200.sh
```

### Ensemble
```
python multi_scale.py
```

### Descriptor Stretching
```
bash descriptor_stretching.sh
```

Finally, the features of all 1,000,000 reference images are saved in ```features/references_byol_multi_scale.hdf5```, and the features of all 50,000 query images are saved in ```features/query_byol_detection_multi_scale_ds.hdf5```. By using the following script, you will get the final submission.
```
import h5py
import numpy as np
from isc.io import read_descriptors, write_hdf5_descriptors

name_r, M_ref = read_descriptors(['features/references_byol_multi_scale.hdf5'])
name_q, M_query = read_descriptors(['features/query_byol_detection_multi_scale_ds.hdf5'])

qry_ids = ['Q' + str(x).zfill(5) for x in range(50_000)]
ref_ids = ['R' + str(x).zfill(6) for x in range(1_000_000)]

out = './submit_track2_visionforce.h5'
with h5py.File(out, "w") as f:
    f.create_dataset("query", data=M_query)
    f.create_dataset("reference", data=M_ref)
    f.create_dataset('query_ids', data=qry_ids)
    f.create_dataset('reference_ids', data=ref_ids)
```



Congratulations!

# One more thing
Because we have three times to submit the results, you can try three different parameters in ```descriptor_stretching.sh```. The ```--factor``` parameter will be ```-2.0```, ```-2.5```, ```-3.0```. 



