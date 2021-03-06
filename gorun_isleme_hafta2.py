#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
os.getcwd()
os.listdir()


# In[17]:


os.getcwd()
os.listdir()
path = os.getcwd()
jpg_files = [f for f in os.listdir(path) 
if f.endswith('.jpg')]


# In[20]:


import numpy as np
import matplotlib.pyplot as plt
image_1 = plt.imread('canakkale.jpg')


# In[23]:


def get_0_1_value_from_triple(temp_1):
    #temp_1=im_1[0,0,:]
    temp = int(temp_1[0]/3+temp_1[1]/3+temp_1[2]/3)
    if temp<110:
        return 0
    else:
        return 1   
get_0_1_value_from_triple(image_1[10,10,:])


# In[37]:



def convert_rgb_to_bw(image_1):
    m,n,k = image_1.shape
    new_image=np.zeros((m,n),dtype='uint8')
    for i in range(m):
        for j in range(n):            
            s = get_0_1_value_from_triple(image_1[i,j,:])
            new_image[i,j] = s
    return new_image


# In[39]:


def get_value_from_triple():
    #temp_1=im_1[0,0,:]
    return int(temp_1[0]/3+temp_1[1]/3+temp_1[2]/3)
get_value_from_triple(image_1[10,10,:])
    


# In[29]:


def conver_rgb_to_gray(im_1):
    m,n,k = im_1.shape
    new_image=np.zeros((m,n),dtype='uint8')
    for i in range(m):
        for j in range(n):
            s = get_value_from_triple(im_1[i,j,:])
            new_image[i,j] = s
    return new_image
        


# In[35]:


im_1 = plt.imread(jpg_files[0])
im_1_gray = conver_rgb_to_gray(im_1)
im_1_bw = conver_rgb_to_bw(im_1)


# In[36]:


plt.subplot(1,3,1)
plt.imshow(im_1)

plt.subplot(1,3,2)
plt.imshow(im_1_gray,cmap='gray')

plt.subplot(1,3,3)
plt.imshow(im_1_bw,cmap='gray')

plt.show()


# In[ ]:




