# -*- coding:utf-8 -*-
'''
this is the enterance of this project
'''

import tensorflow as tf
import os
from model2 import model2
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from project_config import *

if __name__ =='__main__':
    print(" beigin...")
    model = model2(learning_rate,keep_prob,batch_size,40)
    model.inference(normalazation_output_path,test_path,1,True)

#21274





