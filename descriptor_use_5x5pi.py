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

def get_harris_points(harrisim,min_dist= 5,threshold=0.2):
   
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
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
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


def get_pixel(coords,img,half_size=2):
    height = 2*half_size+1
    length = len(coords)
    desp = np.zeros((length,height, height))
    for p in range(0,len(coords)):
        mid_x = coords[p][0]
        mid_y = coords[p][1]
        try:
            img_cut = img[mid_x-2:mid_x+3,mid_y-2:mid_y+3]
            desp[p] = img_cut
        except:
            pass
    return desp

def match_2desp(dp1,dp2,fc1,fc2):
    #spatial.distance.cdist(sim[0].reshape((1, 2)), sim[1].reshape((1, 2)), metric='cosine')
    # create a list
    matchcoord= []
    mid_coord =[]
    min_p2p    = []
    ratio_test = []
    n = 0
    tt_min2 = 0
    for l in tqdm(range(0,len(dp1))):
        i = dp1[l]
        
        min1= 999999
        min2= 0
        for j in range(0,len(dp2)):
            
            o = dp2[j]
            dis = dist(i,o)
            if min1> dis:
                min2 = min1
                min1 = dis
                mid_coord = [fc1[l],fc2[j],1]
                #注意是先Y后X

        
        tt_min2 += min2
        min_p2p.append(min1)
        r_t = min1/min2
        if r_t <= 0.5:
            mid_coord.append(float(r_t))
            matchcoord.append(mid_coord)
            
            #矩阵相似度匹配还没写
            #加入数组后找最大
    return matchcoord

def cosine_Matrix(_matrixA, _matrixB):
     
    _matrixA_matrixB = np.dot(_matrixA, _matrixB.transpose())
    ### 按行求和，生成一个列向量
    ### 即各行向量的模
    _matrixA_norm = numpy.sqrt(numpy.multiply(_matrixA,_matrixA).sum(axis=1))
    _matrixB_norm = numpy.sqrt(numpy.multiply(_matrixB,_matrixB).sum(axis=1))
    return numpy.divide(_matrixA_matrixB, _matrixA_norm * _matrixB_norm.transpose())
def cosine(_vec1, _vec2):
    import numpy
    return float(numpy.sum(_vec1*_vec2))/(numpy.linalg.norm(_vec1)*numpy.linalg.norm(_vec2))

def dist(vec1,vec2):
    import numpy
    vec1 = vec1.ravel()
    vec2 = vec2.ravel()
    dis = numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))

    return dis
    


im1 = array(Image.open('yosemite1.jpg').convert('L'))
im2 = array(Image.open('yosemite2.jpg').convert('L'))
harrisim1 =compute_harris_response(im1)
harrisim2 = compute_harris_response(im2)
filtered_coords1 = get_harris_points(harrisim1)
filtered_coords2 = get_harris_points(harrisim2)


desp1 = get_pixel(filtered_coords1,im1)
desp2 = get_pixel(filtered_coords2,im2)
#%%
# 有关局部极大值，找个卷积卷一下就好

re = match_2desp(desp1,desp2,filtered_coords1,filtered_coords2)
im_re = show_desp('yosemite1.jpg','yosemite1.jpg',re)

cv2.imwrite('re5.png',im_re)
np.savetxt('foo11_5.csv',re,delimiter=',', fmt = '%s')