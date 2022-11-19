# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:33:52 2022

@author: james
"""

import pandas  as pd
import numpy as np
import re
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def get_crp(a):
    a = re.findall("\d+\.\d+|\d+", str(a))
    a = np.array([int(a[1]),int(a[0]),1]) #注意是先y后x
    tr  = np.array([[1.065366 ,-0.001337 ,-299.163870] ,[0.027334 ,1.046342 ,-11.093753],[0.000101,0.000002,1]])
    p = np.dot(tr,a)
    p = p.astype(int)
    a = a.astype(int)
    return p,a

def dist(vec1,vec2):
    import numpy
   # vec1 = vec1.ravel()
   # vec2 = vec2.ravel()
    dis = numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))
   # dis = numpy.sum(numpy.square(vec1 - vec2))
    return dis



data = pd.read_csv('foo11.csv')


prep = 0;
p =0;
y_label = []
ratio = []
for i in data.iterrows():
    
    
    pre,cp = get_crp(i[1][0])
    _,ac  = get_crp(i[1][1])
    r_t   = float(i[1][3])
    dis = dist(ac,pre)
    if dis <= 30:
        y_label.append(2)
    else:
        y_label.append(0)
    ratio.append(r_t)
#%%

tpr, fpr, thersholds = roc_curve(y_label, ratio,pos_label=2)
roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr, 'k--', label='mops ROC (area = {0:.2f})'.format(roc_auc), lw=2,color='r')


data = pd.read_csv('foo11_5.csv')


prep = 0;
p =0;
y_label = []
ratio = []
for i in data.iterrows():
    
    
    pre,cp = get_crp(i[1][0])
    _,ac  = get_crp(i[1][1])
    r_t   = float(i[1][3])
    dis = dist(ac,pre)
    if dis <= 30:
        y_label.append(2)
    else:
        y_label.append(0)
    ratio.append(r_t)

tpr, fpr, thersholds = roc_curve(y_label, ratio,pos_label=2)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='5pi ROC (area = {0:.2f})'.format(roc_auc), lw=2,color='b')
plt.xlim([-0.05, 1.05])  
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


#%%


