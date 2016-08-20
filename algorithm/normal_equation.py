#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Algorithm 4

Normal equation compute theta
theta = feature_normalize(x)
a. x is m*(n+1) matrix
   set x0 = 1
   [1, x1, ... xn]
   [1, x1, ... xn]
        ......
   [1, x1, ... xn]
   [1, x1, ... xn]

b. y is 1*m matrix
   [y1, y2, ... ym]

attention:
1). 我们使用pinv求解矩阵的逆, 当矩阵为奇异矩阵的时候求出的逆矩阵维'伪逆'
2). pinv和inv求出的结果有差异

return theta 1*(n+1) vector
"""
import os
import sys
import numpy as np

def normal_equation(x, y):
    return (np.linalg.pinv(x.getT() * x) * (x.getT() * y.getT())).getT()
