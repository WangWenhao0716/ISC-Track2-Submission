CUDA_VISIBLE_DEVICES=0 python extract_feature_matrix.py \
      --file_list list_files/references \
      --image_dir /dev/shm/references_images \
      --o features/reference_byol_400.hdf5 \
      --model 50_matrix  --GeM_p 3 --checkpoint baseline_matrix_baro_c24.pth.tar --imsize 400
