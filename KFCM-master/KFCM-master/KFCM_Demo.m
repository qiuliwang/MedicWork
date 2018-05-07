clc;clear all;close all;

Img = imread('IMG-0002-00001.jpg');
Img = imresize(Img,0.25)
% Img = imread('noise.bmp');
figure(1),imshow(Img,[]);
Img=rgb2gray(Img);
%参数初始化  
cluster_n =6;                               %聚类数目
m=2;                                        %隶属度权重参数m
iter_max=500;                               %最大迭代次数
e=1e-5;                                     %停止阈值条件
delta=10;                                   %高斯核的delta值  5~15

Vinit=KFCM_init(Img,cluster_n);              %聚类中心初始化
Img_label=KFCM_Img(Img,Vinit,cluster_n,m,iter_max,e,delta);

figure(2);
imshow(Img_label,[]);

figure(3);imshow(Img_label==1);
figure(4);imshow(Img_label==2);
figure(5);imshow(Img_label==3);
figure(6);imshow(Img_label==4);
