CUDA_VISIBLE_DEVICES=0 python extract_feature_matrix.py \
      --file_list list_files/references \
      --image_dir /dev/shm/reference_images \
      --o features/references_byol_200.hdf5 \
      --model 50_matrix  --GeM_p 3 --checkpoint baseline_matrix_baro_c24.pth.tar --imsize 200
