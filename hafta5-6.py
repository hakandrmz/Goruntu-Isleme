#!/usr/bin/env python
# coding: utf-8

# In[1]:


# intensity transformation


# In[3]:


import os
os.getcwd(),os.listdir()


# In[19]:


path=r"C:\Users\Hakan"
file_name_with_path=path+"\canakkale2.jpg"
file_name_with_path


# In[20]:


import matplotlib.pyplot as plt
import numpy as np


# In[38]:


img_0=plt.imread(file_name_with_path)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(img_0)


# In[23]:


np.min(img_0), np.max(img_0)


# In[22]:


img_0.ndim,img_0.shape


# In[30]:


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


# In[31]:


def my_f_1(a,b):
    assert a>0;"intensity positive","error intensity not positive"
    if(a<=255-b):
        return a+b
    else:
        return 255
my_f_1(244,30)


# In[39]:


def my_f_2(a):
    return int(255 - a)
my_f_2(243)


# In[33]:


img_1=convert_rgb_to_gray_level(img_0)
plt.imshow(img_1,cmap='gray')
plt.show()


# In[42]:



m, n = img_1.shape
img_2 = np.zeros((m,n), dtype="uint8")


# In[41]:


for i in range(m):
    for j in range(n):
        intensity = img_1[i,j]
        img_2[i,j] = my_f_2(intensity)


# In[44]:


plt.subplot(2,2,1), plt.imshow(img_0, cmap = 'gray')
plt.subplot(2,2,2), plt.imshow(img_1, cmap = 'gray')
plt.subplot(2,2,3), plt.imshow(img_2, cmap = 'gray')


# In[45]:



x = np.array(list(range(100)))
# y = np.array(list(range(100)))
# y = np.sin(np.array(list(range(100))))
# y = 1 / (1 + np.exp(x))
y1 = np.power(x / float(np.max(x)), 1)
y2 = np.power(x / float(np.max(x)), 10)
y3 = np.power(x / float(np.max(x)), 1/10)



plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)


# In[46]:


def my_f_3(image_001, gamma):
    return np.power(image_001 / float(np.max(image_001)), gamma)


x = img_0
img_100 = np.power(x / float(np.max(x)), 1/10)
plt.imshow(img_100)


# In[ ]:




