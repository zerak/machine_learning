#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalizes the features in x
x = feature_normalize(x)
a. x is m*n matrix
   [x1, ... xn]
   [x1, ... xn]
        ......
   [x1, ... xn]
   [x1, ... xn]

return x

attention:
1. 为了使得特征向量矩阵每个元素为double类型
   必须对矩阵x的每个元素乘以1.0，否则当x的每个元素都是int类型时对x[i]直接赋值会导致小数点被丢失
2. 为了使得计算的精度更准确, 求解矩阵的标准差的时候 需要把函数std的参赛ddof设为1
   同时也为了和octave的结果保持一致
"""
import os
import sys
import numpy as np

def feature_normalize(x):
    x = x * 1.0
    mean_x = x.mean(0)
    std_x = x.std(0, ddof=1)
    # number of training examples
    m = (x.shape)[0]
    for i in xrange(m):
        x[i] = (x[i]-mean_x)/std_x
    return x
