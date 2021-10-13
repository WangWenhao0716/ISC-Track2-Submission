CUDA_VISIBLE_DEVICES=0,1 python descriptor_stretching.py \
    --query_descs features/query_byol_multi_scale.hdf5 \
    --db_descs features/references_byol_multi_scale.hdf5 \
    --train_descs features/train_byol_multi_scale.hdf5 \
    --factor -2.5 --n 5 \
    --o features/temp.csv \
    --reduction avg
