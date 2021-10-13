from isc.io import read_descriptors, write_hdf5_descriptors
import numpy as np
from sklearn.preprocessing import normalize

scales = [200, 256, 320, 400]
path = './features/'
features_r = []
for s in scales:
    file_r = [path + 'references_' + str(i) + '_byol_' + str(s) + '.hdf5' for i in range(20)]
    name_r, feature_r = read_descriptors(file_r)
    features_r.append(feature_r)

features_r_multi = normalize(sum(features_r)/len(features_r))
write_hdf5_descriptors(features_r_multi,name_r, path + 'references_byol_' + str(scales) +'.hdf5')

features_t = []
for s in scales:
    file_t = [path + 'train_' + str(i) + '_byol_' + str(s) + '.hdf5' for i in range(20)]
    name_t, feature_t = read_descriptors(file_t)
    features_t.append(feature_t)

features_t_multi = normalize(sum(features_t)/len(features_t))
write_hdf5_descriptors(features_t_multi,name_t, path + 'train_byol_' + str(scales) +'.hdf5')

features_q = []
for s in scales:
    file_q = [path + 'query_' + str(i) + '_byol_' + str(s) + '_detection.hdf5' for i in range(1)]
    name_q, feature_q = read_descriptors(file_q)
    features_q.append(feature_q)

features_q_multi = normalize(sum(features_q)/len(features_q))
write_hdf5_descriptors(features_q_multi,name_q, path + 'query_byol_detection_' + str(scales) +'.hdf5')

