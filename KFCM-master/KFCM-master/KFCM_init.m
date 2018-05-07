function [Vinit]=KFCM_init(Img,cluster_n)
% KFCM初始化函数，得到初始聚类中心，使用K-means算法进行初始化
% 输入参数：
%         Img--原图像
%         cluster_n--聚类数
% 输出参数：
%         Vinit--初始聚类中心
%
%by--zou


Img=double(Img);
[row,col]=size(Img);
Imgtmp=reshape(Img,row*col,1);
[index,Vinit]=kmeans(Imgtmp,cluster_n,'emptyaction','singleton');
