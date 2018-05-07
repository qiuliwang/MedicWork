
# coding: utf-8

'''
Created by WangQL

py file for 3D conv

Medic
'''
# In[14]:


import dicom # read the dicom file
import os # do directory operations
import pandas as pd # nice for data analysis

data_dir = 'data\\'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('test_labels.csv', index_col = 0)
print(labels_df.head())


# In[15]:

for patient in patients:
    label = labels_df.at[patient, 'cancer']
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x : float(x.ImagePositionPatient[2]))
    print(len(slices), slices[0].pixel_array.shape)

print(len(patients))


# In[16]:


len(patients)


# In[17]:


import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

IMG_PX_SIZE = 50
HM_SLICES = 20

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]
        
def mean(l):
    return sum(l) / len(l)

def process_data(patient, labels_df, img_px_size = 50, hm_slices=20, visualize=False):
    label = labels_df.at[patient, 'cancer']
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s, force = True) for s in os.listdir(path)]
    slices.sort(key = lambda x : float(x.ImagePositionPatient[2]))
    
    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE, IMG_PX_SIZE)) for each_slice in slices]
    chunk_size = math.ceil(len(slices) / HM_SLICES)
    
    for slice_chunk in chunks(slices, chunk_size):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)
        
    if len(new_slices) == HM_SLICES - 1:
        new_slices.append(slice_chunk)
    
    if len(new_slices) == HM_SLICES + 2:
        new_val = list(map(mean, zip(*[new_slices[HM_SLICES - 1], new_slices[HM_SLICES]])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES - 1] = new_val
        
    if len(new_slices) == HM_SLICES + 1:
        new_val = list(map(mean, zip(*[new_slices[HM_SLICES - 1], new_slices[HM_SLICES]])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES - 1] = new_val
        
    print(len(new_slices))
#     for slice in new_slices:
#         plt.imshow(slice)
#         plt.show()
        
    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(slices[:20]):
            y = fig.add_subplot(4, 5, num + 1)
    #         new_image = cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE, IMG_PX_SIZE))
    #         y.imshow(new_image, cmap = 'gray')
            y.imshow(each_slice)

    if label == 1 : 
        label = np.array([0,1])
    elif label == 1 : 
        label = np.array([1,0])
    
    return np.array(new_slices), label

much_data = []

for num, patient in enumerate(patients):
    if num%100 == 0:
        print(num)
    try:
        img_data, label = process_data(patient, labels_df, img_px_size = IMG_PX_SIZE, hm_slices = HM_SLICES)
        much_data.append([img_data, label])
    except KeyError as e:
        print('This is unlabeled data')
        
np.save('muchdata-{}-{}-{}.npy'.format(IMG_PX_SIZE, IMG_PX_SIZE, HM_SLICES), much_data)


# In[1]:


import tensorflow as tf
import numpy as np

IMG_SIZE_PX = 50
SLICE_COUNT = 20
n_classes = 2

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides = [1,1,1,1,1], padding = 'SAME')

def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize = [1, 2, 2, 2, 1], strides = [1, 2, 2, 2, 1], padding = 'SAME')

def convolutional_neural_network(x):
    weights = {
        'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
        'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
        'W_fc': tf.Variable(tf.random_normal([54080, 1024])),
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }
    
    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    x = tf.reshape(x, shape = [-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])
    
    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)
    
    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2'] + biases['b_conv2']))
    conv2 = maxpool3d(conv2)
    
    fc = tf.reshape(conv2, [-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)
    
    output = tf.matmul(fc, weights['out'] + biases['out'])
    
    return output


def train_neural_network(x):
    
    much_data = np.load('muchdata-50-50-20.npy')
    train_data = much_data[:10]
    
    validation_data = much_data[-10:]
    
    
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 5
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            success_total = 0
            attempt_total = 0
            for data in train_data:
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict = {x: X, y: Y})
                    epoch_loss += c
                except Exception as e:
                    pass
                
            print('Epoch ', epoch, ' completed out of ', hm_epochs, ' loss: ', epoch_loss)
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accrracy: ', accuracy.eval({x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))
    
    
train_neural_network(x)

