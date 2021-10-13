CUDA_VISIBLE_DEVICES=0 python extract_feature_matrix.py \
      --file_list list_files/dev_queries \
      --image_dir /dev/shm/query_images_detection \
      --o features/query_byol_200_detection.hdf5 \
      --model 50_matrix  --GeM_p 3 --checkpoint baseline_matrix_baro_c24.pth.tar --imsize 200
