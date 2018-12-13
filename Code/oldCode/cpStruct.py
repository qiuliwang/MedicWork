'''
copy file structure
WangQL
12/22/2017
'''
import os
import sys
source = os.path.realpath("C:\\Users\\WangQL\\Documents\\GitCode\\PersonalWork\\DicomWork\\Data\\DOI") 
target = os.path.realpath("C:\\Users\\WangQL\\Documents\\GitCode\\PersonalWork\\DicomWork\\Data\\DOI JPG")

print("\nCopy directory structure")
if len(sys.argv)==1: #没有命令行，提示输入
        print()
        sourceRoot=source #来源目录
        destRoot=target
 ##目标目录
#        sourceRoot='c:\\fpc2.6'
#        destRoot='f:\\test'
else: #支持命令行
        sourceRoot=argv[1]
        destRoot=argv[2]
        print("\tFrom directory: '",sourceRoot,"'")
        print("\tTo directory: '",destRoot,"'.")

#核查来源/目标目录是否存在
if not os.path.isdir(sourceRoot):
        print('Not found source directory:',sourceRoot)
        exit()
if not os.path.isdir(destRoot):
        print('Not found dest directory:',destRoot)
        exit()

#复制目录结构
for dirname,dirs,files in os.walk(sourceRoot):
##        print(dirname,':')
##        print(dirs)
        dirTemp=dirname.replace(sourceRoot,destRoot)
        if len(dirs)!=0: # 非空目录
                print(dirTemp)
                os.chdir(dirTemp)
                for s in dirs:
                        os.makedirs(s,0o777,True) #True屏蔽目录已经存在错误提示
                        print('\t',s) #正在建立的子目录
                print('-'*50)
        
os.chdir(destRoot) #环境清理


# '''
# https://www.cnblogs.com/iforever/p/4313704.html
# 12.22.2017
# '''
# import os
# import sys

# # source = os.path.realpath(sys.argv[1]) 
# # target = os.path.realpath(sys.argv[2])
# source = os.path.realpath("C:\\Users\\WangQL\\Documents\\GitCode\\PersonalWork\\DicomWork\\Data\\SPIE-AAPM Lung CT Challenge") 
# target = os.path.realpath("C:\\Users\\WangQL\\Documents\\GitCode\\PersonalWork\\DicomWork\\Data\\SPIE-AAPM Lung CT Challenge JPG")

# def isdir(x):
#     return os.path.isdir(x) and x != '.svn'
# def mkfloders(src,tar):
#     paths = os.listdir(src)
#     paths = map(lambda name:os.path.join(src,name),paths)
#     paths = filter(isdir, paths)
#     if(len(paths)<=0):
#         return
#     for i in paths:
#         (filepath, filename)=os.path.split(i)
#         targetpath = os.path.join(tar,filename)
#         not os.path.isdir(targetpath) and os.mkdir(targetpath)
#         mkfloders(i,targetpath)

# if __name__=="__main__":
#     if(os.path.isdir(source)):
#         if(target.find(source) == 0):
#             print("cannot create files")
#         else:
#             if not os.path.isdir(target):
#                 os.mkdir(target)
#             mkfloders(source,target)
#     else:
#         print("no source")