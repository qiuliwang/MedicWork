function Img_label=KFCM_Img(Img,Vinit,cluster_n,m,iter_max,e,delta)
% FCM(模糊C均值)算法
% 输入参数：
%     Img--原图像
%     cluster_n--聚类数
%     m--隶属度程度（default=2）
%     iter_max--最大迭代次数（default=1000）
%     e--停止阈值（default=1e-5）
% 输出参数：
%     Img_label--图像聚类标签（1.2.3...）

if nargin==3
    m=2;iter_max=1000;e=1e-5;delta=10;
end
if nargin==4
    iter_max=1000;e=1e-5;delta=10;
end
if nargin==5
    e=1e-5;delta=10;
end
if nargin==6
    delta=10;
end
    
Img=double(Img);
[nrow,ncol]=size(Img);

%参数初始化  
Img_reshape=reshape(Img,nrow*ncol,1);       %源数据
options = [m;iter_max;e;1;delta]; 

%开始聚类
[center,U,obj_fcn] = kfcm(Img_reshape,Vinit,options); 

%标号和复原图像
maxU = max(U);
maxUn=repmat(maxU,cluster_n,1);
U=U';
maxUn=maxUn';
One_index=(maxUn==U);

Img_label=zeros(size(Img_reshape));
for i=1:cluster_n
    index=floor(find(One_index(:,i)==1));
    Img_label(index)=i;
end

Img_label=reshape(Img_label,nrow,ncol);

%根据灰度级进行过排序
tmp=zeros(1,cluster_n);
for i=1:cluster_n
    tmp(i)=mean(Img(Img_label==i));
end 
[data,index]=sort(tmp);

Img_label0=zeros(size(Img_label));
for i=1:cluster_n
    Img_label0(Img_label==index(i))=i;
end

Img_label=Img_label0;
