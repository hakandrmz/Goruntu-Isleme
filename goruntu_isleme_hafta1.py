#!/usr/bin/env python
# coding: utf-8

# In[51]:


import os
import numpy as np
import matplotlib.pyplot as plt


# In[52]:


def compare_list_ndarray():
    list_1 = [1,2,3,4,"2Merhaba,3",'4',5,6]
    list_2 = [2,"3Merhaba,3",'1123',15,26]
    list_1 + list_2 

    list_1 = [1,2,3,4]
    list_2 = [1,2,3,4]
    list_1 + list_2 +[10]

    list_3=np.asarray([1,2,3,4])  #ndarray asarray
    list_4=np.asarray([1,2,3,4])
    print(list_3 + list_4 +10)
compare_list_ndarray()
def get_jpeg_files():
    os.getcwd()
    os.listdir()
    path = os.getcwd()
    jpg_files = [f for f in os.listdir(path) 
    if f.endswith('.jpg')]
    return jpg_files
print(get_jpeg_files())


# In[105]:


def my_rotate(im_1): 
    m,n,k = im_1.shape
    new_image =  np.zeros((n,m,k),dtype='uint8')
    for i in range(m):
        for j in range(n):
            temp = image_1[i,j]
            new_image[j,i]=temp
    return new_image


# In[106]:


def display_two_image(im_1,im_2):
    plt.subplot(1,2,1)
    plt.imshow(image_1)

    plt.subplot(1,2,2)
    plt.imshow(image_2+30)

    plt.show()


# In[107]:


image_1 = plt.imread('canakkale.jpg')


# In[110]:


image_2 = my_rotate(image_1)
display_two_image(image_1,image_2)

