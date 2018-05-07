import  matplotlib.pyplot as plt
import  matplotlib.image as mpimg
import  numpy as np
import  cv2
import tensorflow
drawing = False #鼠标按下为真
mode = True #如果为真，画矩形，按m切换为曲线
ix,iy=-1,-1
px,py=-1,-1
x1,y1=-1,-1
x2,y2=-1,-1

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def Fill(i, j,img):
    stack=[]
    Seed=[i,j]
    xr=0
    xl=0
    need=False
    old=img[i,j]
    stack.append(Seed)
    while(len(stack)!=0):
        Seed=stack.pop()
        x,y=Seed
        while(img[x][y]==old and x<img.shape[0]-1):
            img[x][y]=256
            x+=1
        xr = x-1
        x,y=Seed
        x=x-1
        while(img[x][y]==old and x<img.shape[0]-1):
            img[x][y]=256
            x-=1
        xl=x+1
        x=xl
        y=y+1
        while(x<=xr):
            need=False
            while(img[x][y]==old and x<img.shape[0]-1):
                need=True
                x+=1
            if(need):
                Seed=[x-1,y]
                stack.append(Seed)
                need=False
            while(img[x,y]!=old and x<=xr):
                x+=1
        x=xl
        y=y-2
        while (x <= xr):
            need = False
            while (img[x][y] == old and x<img.shape[0]-1):
                need = True
                x += 1
                if(x<0):break
            if (need):
                Seed=[x-1,y]
                stack.append(Seed)
                need = False
            while (img[x, y] != old and x <= xr):
                x += 1

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,px,py,x1,x2,y1,y2

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy=x,y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(image,(ix,iy),(x,y),(0,0,255),0)
        px,py=-1,-1
        x1=ix
        y1=iy
        x2=x
        y2=y

def FCM(image,mask,clusterNum = 10,expo = 2,max_inter=300,min_impro = 1e-10):
    gray = rgb2gray(image)
    gray = gray[mask[1]:mask[3], mask[0]:mask[2]]
    [row, col] = gray.shape
    data = np.array(gray, dtype='double')
    clusterNum = 5
    expo = 2
    max_inter = 300
    min_impro = 1e-10
    obj_fcn = np.zeros([max_inter, 1])
    Upre = np.zeros([row, col, clusterNum])
    center = np.zeros([clusterNum, 1])
    Upre = abs(np.random.randn(row, col, clusterNum))
    Upre_Sum = Upre.sum(axis=1).sum(axis=0)
    Upre = Upre / Upre_Sum
    for Iter in range(0, max_inter):
        for i in range(0, clusterNum):
            Up = (Upre[:, :, i] * data[:, :]).sum(axis=1).sum(axis=0)
            Down = (Upre[:, :, i]).sum(axis=1).sum(axis=0)
            center[i, 0] = Up / Down
        out = np.zeros([row, col, clusterNum])
        for i in range(0, clusterNum):
            out[:, :, i] = abs(data[:, :] - center[i, 0])
            obj_fcn[i] = obj_fcn[i] + ((Upre[:, :, i] ** expo) * (out[:, :, i] ** 2)).sum(axis=1).sum(axis=0)
        for i in range(0, clusterNum):
            top = 0
            for j in range(0, clusterNum):
                top = (out[:, :, i] / out[:, :, j]) ** (expo - 1)
            Upre[:, :, i] = 1 / top
        for i in range(0, clusterNum):
            Upre_Sum = Upre.sum(axis=1).sum(axis=0)
            Upre = Upre / Upre_Sum
        if (Iter > 1):
            print(abs(obj_fcn[Iter] - obj_fcn[Iter - 1]))
            if (abs(obj_fcn[Iter] - obj_fcn[Iter - 1] <= min_impro)):
                break
    newing = np.zeros([row, col])
    for i in range(0, row):
        for j in range(0, col):
            MaxU = Upre[i, j, 0]
            index = 0
            for k in range(0, clusterNum):
                if (Upre[i, j, k] > MaxU):
                    MaxU = Upre[i, j, k]
                    index = k
            newing[i, j] = round(255 * (1 - (index) / (clusterNum)))
    return newing;

def getTurmor(image,data):
    [row,col]=data.shape
    row1=int(row/2)
    col1=int(col/2)
    Fill(row1, col1, data)
    for i in range(0, row):
        for j in range(0, col):
            if (data[i, j] != 256):
                image[i, j, :] = 0
    return image

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

image = mpimg.imread('2.jpg')
while(1):
    cv2.imshow('image', image)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('n') :
        print([x1,y1,x2,y2])
        mask=[x1,y1,x2,y2]
        break
    elif k == 27:
        break
cv2.destroyAllWindows()


new_image=FCM(image,mask,3)
plt.imshow(new_image,cmap=plt.cm.gray)
plt.axis("on")
plt.show()

image=image[mask[1]:mask[3], mask[0]:mask[2]]
image=getTurmor(image,new_image)
plt.imshow(image,cmap=plt.cm.gray)
plt.axis("on")
plt.show()

# 腐蚀膨胀算子