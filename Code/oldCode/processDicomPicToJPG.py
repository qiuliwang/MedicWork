'''
Created by WangQL
12/22/2017
'''

import readfiles
import optForOne

# get dicom dir and set the dir for jpgs
windir = "C:\\Users\\WangQL\\Documents\\GitCode\\PersonalWork\\DicomWork\\Data\\DOI"
filelist, jpglist = readfiles.get_two_dirs(windir)
# print(filelist[0])
# print(jpglist[0])

if(len(filelist) != len(jpglist)):
    print("error!")
else:
    count = len(filelist)
    print(count)
    for i in range(0, count):
        print(i / count * 100)
        optForOne.showPic(filelist[i], jpglist[i])
