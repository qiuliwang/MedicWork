import os
import tensorflow as tf

base_dir = "/data0/LUNA/"
base_dir1 = "/data1/LUNA/"
annatation_file = base_dir + 'CSVFILES/annotations.csv'
candidate_file =base_dir +  'CSVFILES/candidates_V2.csv'
plot_output_path = base_dir + 'cubic_npy'
plot_output_path1 = base_dir + 'cubic_npy_9'
if not os.path.exists(plot_output_path):
    os.mkdir(plot_output_path)
if not os.path.exists(plot_output_path1):
    os.mkdir(plot_output_path1)

normalazation_output_path = base_dir +'cubic_normalization_npy'
if not os.path.exists(normalazation_output_path):
    os.mkdir(normalazation_output_path)
test_path = base_dir + 'cubic_normalization_test/'
if not os.path.exists(test_path):
    os.mkdir(test_path)

batch_size = 200


keep_prob = 0.7