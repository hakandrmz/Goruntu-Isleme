#!/usr/bin/env python
# coding: utf-8

# In[68]:



def get_mask_for_edge():
    return np.array([-1,0,1,-2,0,2,-1,0,1]).reshape(3,3)

def apply_mask_for_edge(part_of_image):
    mask=get_mask_for_edge()
    return sum(sum(part_of_image*mask))
 
def get_edges(im_1):
    m=im_1.shape[0]
    n=im_1.shape[1]
    
    im_2 = np.zeros((m,n))
    
    for i in range(3,m-3):
        for j in range(3,n-3):
            poi = im_1[i-1:i+2,j-1:j+2]
            
            im_2[i,j] = apply_mask_for_edge(poi)
    return im_2

poi_1 = get_mask_for_edge()
apply_mask_for_edge(poi_1)

    


# In[69]:


im_with_edges = get_edges(im_2)


# In[70]:


plt.imshow(im_with_edges,cmap='gray')


# In[71]:


plt.figure(figsize=(20,20))
plt.subplot(1,3,1),plt.imshow(im_1)
plt.subplot(1,3,2),plt.imshow(im_2,cmap='gray')
plt.subplot(1,3,3),plt.imshow(im_with_edges,cmap='gray')


# In[75]:


file_path=r"C:\Users\Hakan\camera.png"
im_1=plt.imread(file_path)
im_2=convert_rgb_to_gray_level(im_1)
plt.figure(figsize=(20,20))
plt.subplot(1,2,1),plt.imshow(im_1)
plt.subplot(1,2,2),plt.imshow(im_2,cmap='gray')
plt.show()


# In[76]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg


# In[74]:


def convert_rgb_to_gray_level(im_1):
    m = im_1.shape[0]
    n = im_1.shape[1]
    im_2 = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            im_2[i,j] = get_distance(im_1[i,j,:])
    return im_2


def get_distance(v, w = [1/3, 1/3, 1/3]):
    a, b, c = v[0], v[1], v[2]
    w1, w2, w3 = w[0], w[1], w[2]
    d = ((a**2) * w1 + (b**2) * w2 + (c**2) * w3)**.5
    # d = (a*w1)**2 + (b*w2)**2 + (c*w3)**2)**.5
    return d



