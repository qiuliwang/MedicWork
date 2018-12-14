#/data0/LUNA/cubic_normalization_npy
import os
from data_prepare import get_train_batch,get_all_filename,get_test_batch
import random
import tensorflow


# file_list = []
# dir1 = "/data0/LUNA/cubic_normalization_npy/"
# dir2 = "/data0/LUNA/cubic_normalization_test/"

# all_test_filenames = get_all_filename(dir2, 40)
# #print(all_test_filenames[:100])
# random.shuffle(all_test_filenames)
# test_batch, batch_label = get_test_batch(all_test_filenames)

# print(batch_label)

# file_list = os.listdir(dir)
# print(len(file_list))

# real_list = []
# fake_list = []

# for item in file_list:
#     if 'fake' in item:
#         fake_list.append(item)
#     if 'real' in item:
#         real_list.append(item)

# print(len(real_list))
# print(len(fake_list))