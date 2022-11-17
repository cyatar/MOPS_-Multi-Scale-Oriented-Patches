# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:30:31 2022

@author: james
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 22:28:08 2022

@author: james
"""


from PIL import Image
from numpy import *

from pylab import *
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import numpy as np

from numpy import *

from time import sleep

from tqdm import tqdm


def compute_harris_response(im,sigma =3 ):
    
    
    #CCD
    imx = zeros(im.shape)
    gaussian_filter(im,(sigma,sigma),(0,1),imx)
    imy = zeros(im.shape)
    gaussian_filter(im,(sigma,sigma),(1,0),imy)
    
    
    
    #CCHM
    Wxy= gaussian_filter(imx*imy,sigma)
   
    Wxx = gaussian_filter(imx*imx,sigma)
    Wyy = gaussian_filter(imy*imy,sigma)
    
    
    #CCM
    
    Wdet = Wxx*Wyy - Wxy**2
    Wtr  = Wxx+Wyy
    
    
    return Wdet/Wtr

def get_harris_points(harrisim,min_dist= 20,threshold=0.2):
   
    conner_thershold = harrisim.max()*threshold
    harrism_t =(harrisim>conner_thershold)*1 
    
    #get position
    
    
    coords = array(harrism_t.nonzero()).T
    
    # H recall
    
    candidate_values = [harrisim[c[0],c[1]]for c in coords]
    
    
    # sort 
    index = argsort(candidate_values)
    
    # save to array

    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist]=1
    
    filtered_coords=[]

    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]==1].all():
            
                filtered_coords.append(coords[i])
                allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                (coords[i,1]-min_dist):(coords[i,1]+min_dist)]= 0
    
    return filtered_coords
    # 注意，这里返回的坐标是先Y后X的
    # 即 [1]是X ， [0]是Y

    
    
def plot_harris_points(im,filtered_coords,n = 0):
    image = im.copy()
   # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],"+",color='red')
    plt.savefig("./"+str(n)+".jpg")
    plt.show()
    
def show_desp(path1,path2,matre):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    image = concatenate([img1, img2], axis=1)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
 #   add_len = img1.shape[0]
    
    
    point_size = 1
    point_color = (255, 0, 0) # BGR
    thickness = 4 #  0 、4、8
  #  plt.figure(dpi=40,figsize=(31,12))
    for p in matre:
        
        #是否启用
       
         cv2.circle(image, [p[0][1],p[0][0]], point_size, point_color, thickness)
         cv2.circle(image, [p[1][1]+640,p[1][0]], point_size, point_color, thickness)
         #print(1)
        # print(p[1])
         cv2.line(image,[p[1][1]+640,p[1][0]],[p[0][1],p[0][0]],(0,255,0),1)

        
    
    #plt.imshow(image)
   # plt.axis('off')
    #cv2.imwrite("im3.png",image)
   # plt.show()
    return image
    

    
    
    '''   for i in range(0,len(p1)):
            x = [p1[i][0],re[i][0] + 480 ]
            y = [p1[i][1],re[i][1]]
        
    '''   
def angle_of_vector(v1, v2):
    pi = 3.1415
    
    from math import sqrt, pow, acos

    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    #return (acos(cos) / pi) * 180 # 角度制
    return acos(cos) # 弧度制
    
def get_gra_direction(img,loc):
    import numpy as np
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize = 3)#只计算横轴
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize = 3)# 竖轴
    vec1 = [sobelx[loc[0],loc[1]],sobely[loc[0],loc[1]]]
    vec2 = [1,0]
    out  = angle_of_vector(vec1,vec2)    
    return out

def rotate_cut(im,ang,loc):
    img = im.copy()
    y = loc[0]
    x = loc[1]
    w = 40
    h = 40
    im_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ld = [x-20,y+20]
    lu = [x-20,y-20]
    rd = [x+20,y+20]
    ru = [x+20,y-20]
    m = [(ld[0]+rd[0]) // 2 ,(ld[1]+rd[1]) // 2 ]
     
   # mx = [x+10,y-20]
    lu = Srotate(ang,lu[0],lu[1],x,y)
    ld = Srotate(ang,ld[0],ld[1],x,y)
    ru = Srotate(ang,ru[0],ru[1],x,y)
    rd = Srotate(ang,rd[0],rd[1],x,y)
    m  = Srotate(ang,m[0],m[1],x,y)
   # print(lu)
   # m_x = (ld[0]+rd[0]) // 2 
   # m_y = (ld[1]+rd[1]) // 2 
   
    cv2.line(img, (int(x),int(y)), (int(m[0]),int(m[1])), (0,0,255), 1)
    cv2.line(img, (int(lu[0]),int(lu[1])), (int(ld[0]),int(ld[1])), (0,0,255), 1)
    cv2.line(img, (int(ld[0]),int(ld[1])), (int(rd[0]),int(rd[1])), (0,0,255), 1)
    cv2.line(img, (int(rd[0]),int(rd[1])), (int(ru[0]),int(ru[1])), (0,0,255), 1)
    cv2.line(img, (int(ru[0]),int(ru[1])), (int(lu[0]),int(lu[1])), (0,0,255), 1)
    locate = array([lu,ld,rd,ru])
    
    return img,locate
    
    
def Srotate(angle,valuex,valuey,pointx,pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
    sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
    return int(sRotatex),int(sRotatey)

def polly_mask(locate):
    img = np.zeros((480,640))
    cv2.fillConvexPoly(img, locate ,255)
    img = img.astype(np.uint8)
    #plt.imshow(img)
    return img

def just_cut(im,loc):
    img = im.copy()    
   
    
    
def select_p(poins,min_dist = 20):
    import numpy
    points = poins.copy()
    n = 0
    j = 0
    while n +1 <= len(points):
        p1 =  points[n]
        j = 0
        pp = []
        while  j+1 <= len(points):
            if j == n:
                j += 1
                continue
            else:
                p2 = points[j]
                dis = numpy.sqrt(numpy.sum(numpy.square(p1 - p2)))
                if dis < min_dist:
                    #print(dis)
                    pp.append(j)
                  #  print("PP!")
                j += 1
                
        z = 0
        for i in pp:
            
            points.pop(i-z)
            z += 1
         
        n += 1
    return points

def rotate(image, angle, center=None, scale=1.0): #1
    pi = 3.141592653
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4
    angle = -(angle / pi) * 180 
    M = cv2.getRotationMatrix2D(center, angle, scale) #5
 
    rotated = cv2.warpAffine(image, M, (w, h)) #6
    
    M = cv2.getRotationMatrix2D(center, 90 , scale) #7
 
    rotated = cv2.warpAffine(rotated, M, (w, h)) #8
    return rotated #7

def get_mops(path,se = True):
    im1 = cv2.imread(path)
    # 这里给一个高斯滤完波后的结果
   # im1_gaussianBlur = cv2.GaussianBlur(im1, (3, 3), 1)
    # 或许没有必要高斯滤波？
    im1_RGB = cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)
   # im1_gray = cv2.cvtColor(im1_gaussianBlur,cv2.COLOR_BGR2GRAY)
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    harrisim1 = compute_harris_response(im1_gray)
    filtered_coords1 = get_harris_points(harrisim1)
    
    # 这边进行了有效筛选(降低点密度)
    if se:
        poi = select_p(filtered_coords1) #先y后x
    else:
        poi = filtered_coords1
    # 第一次画图，显示harris角点
    plot_harris_points(im1,poi)
    im_re = im1_RGB.copy()
    
    #创建储存矩形的数组
    all_poi_im = []
    
    #显示旋转矩形,并将对应desc储存到对应数组中
    # 
    
    
    for j in tqdm(range(0,len(poi))):
        try:
            i = poi[j]
            re = get_gra_direction(im1_gray,i)
            #计算梯度
            im_re, loc_a = rotate_cut(im_re,re,i)    
            #计算
            #print(i)
            bounding_boxes = cv2.boundingRect(loc_a)
            [x , y, w, h] = bounding_boxes
            target_im = im1_gray[y:y+h,x:w+x].copy()
            im_out = rotate(target_im,re)
            mid_x = im_out.shape[0]//2
            mid_y = im_out.shape[0]//2
            im_fre = im_out[mid_y-20:mid_y+20,mid_x-20:mid_x+20].copy()
            #final_im = cv2.resize(im_fre,dsize=(8,8),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
            #修改成金字塔下采样
            im_fre = morphology(im_fre)
            
            m = normalize(im_fre)
            m = [i,m]
            all_poi_im.append(m) 
        except:
            pass
    return im_re,all_poi_im

def dist(vec1,vec2):
    import numpy
   # vec1 = vec1.ravel()
   # vec2 = vec2.ravel()
   # dis = numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))
    dis = numpy.sum(numpy.square(vec1 - vec2))
    return dis


def match_2desp(p1,p2,thre = 0.5):
    #spatial.distance.cdist(sim[0].reshape((1, 2)), sim[1].reshape((1, 2)), metric='cosine')
    # create a list
    
    
    matchcoord= []
    mid_coord =[]
    min_p2p    = []
    ratio_test = []
    n = 0
    tt_min2 = 0
    for l in tqdm(range(0,len(p1))):
        i = p1[l]
        
        min1= 999999
        min2= 0
        for j in p2:
            
            
            dis = dist(i[1],j[1])
            if min1> dis:
                min2 = min1
                min1 = dis
                mid_coord = [i[0],j[0],1]
                #注意是先Y后X

        
        tt_min2 += min2
        min_p2p.append(min1)
        r_t = min1/min2
        if r_t <= 0.5:
            mid_coord.append(float(r_t))
            matchcoord.append(mid_coord)
        
        '''if r_t < thre :
            mid_coord[-1] = -1
            mid_coord.append(r_t)
            matchcoord.append(mid_coord)
        else:
            mid_coord.append(r_t)
            matchcoord.append(mid_coord)
        '''
    
 
       
        
        
    return matchcoord

def morphology(im):
    
    img = im.copy()
    for i in range(1,3):
        img = cv2.pyrDown(img)

    img = cv2.resize(img.copy(),(8,8))
    return img

def normalize(pi8):
    
    m = pi8.ravel().astype(float)
    m = m - np.mean(m)
    m = m / np.std(m)
    m = m/np.linalg.norm(m)
    
    return m


path1 = "yosemite1.jpg"
path2 = "yosemite2.jpg"
im1,mp1  = get_mops(path1,False)
im2,mp2  = get_mops(path2,False)

re = match_2desp(mp1,mp2)

im_re = show_desp(path1,path2,re)
plt.axis('off')
plt.imshow(im_re)
im_re = cv2.cvtColor(im_re, cv2.COLOR_BGR2RGB)
cv2.imwrite('re.png',im_re)
np.savetxt('foo1.csv',re,delimiter=',', fmt = '%s')