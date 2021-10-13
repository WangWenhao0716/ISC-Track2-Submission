CUDA_VISIBLE_DEVICES=0 python extract_feature_matrix.py \
      --file_list list_files/train \
      --image_dir /dev/shm/training_images \
      --o features/train_byol_400.hdf5 \
      --model 50_matrix  --GeM_p 3 --checkpoint baseline_matrix_baro_c24.pth.tar --imsize 400 
