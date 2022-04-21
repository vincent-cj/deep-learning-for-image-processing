# -*- coding: utf-8 -*-
"""
Created on 2022/4/3 下午10:23

@Project -> File: deep-learning-for-image-processing -> test_demo.py

@Author: vincent-cj

@Describe:
"""
import torch
from torch import nn


# In[3]:


def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i + h, j:j + w] += X[i, j] * K
    return Y


# In[4]:


X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, K))


temp = torch.zeros(1, 1, 4, 4)
temp[:, :, 1:3, 1:3] = X
tconv = nn.Conv2d(1, 1, kernel_size = 2, bias = False)
tconv.weight.data = torch.tensor(K.numpy()[::-1, ::-1].copy()[None, None])
print(tconv(temp))


# In[5]:


X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print(tconv(X))


# In[6]:


# padding 是应用在output上的，如果padding是1，则在上下左右各减少一行。
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print(tconv(X))


# In[9]:


# stride
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print(tconv(X))


# In[11]:


# 验证卷积和转置卷积的维度
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(tconv(conv(X)).shape == X.shape)


X = torch.arange(9.).reshape(1, 1, 3, 3)
K = torch.tensor([[1., 2.], [3., 4.]])[None, None]
conv = nn.Conv2d(1, 1, kernel_size = 2, padding = 0, stride = 1, bias = False)
conv.weight.data = K
Y = conv(X)
print(Y)


