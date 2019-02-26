path1 = '/home/zhangjiajia/fusion/fusion0904/mali_fpr_data/'
path2 = '/home/wangqiuli/Documents/august_fpr/train/'

import os
import re
import sys
import getpass
import shutil

# sourceSrcDir = '/home/wangqiuli/Documents/august_fpr/train/'
dstSrcDir = '/home/zhangjiajia/fusion/fusion0904/mali_fpr_fusiondata/'
data = os.listdir(dstSrcDir)
for one in data:
	if 'real' in one:
		os.remove(dstSrcDir + one)
        