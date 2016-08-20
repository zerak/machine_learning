#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Algorithm 3

Gradient descent to learn theta
theta = gradient_descent(x, y, theta, alpha)
a. x is m*(n+1) matrix
   set x0 = 1
   [1, x1, ... xn]
   [1, x1, ... xn]
        ......
   [1, x1, ... xn]
   [1, x1, ... xn]
b. y is 1*m matrix
   [y1, y2, ... ym]
c. theta is 1*(n+1) matrix
   [theta0, theta1, ... thetan]
d. alpha is real number

return theta 1*(n+1) vector
"""
import os
import sys
import numpy as np

def gradient_descent(x, y, theta, alpha):
    # number of training examples
    m = (y.shape)[1]
    # must be use copy action
    # if not tmp_theta and theta is same obj
    # change theta, tmp_theta will be change
    tmp_theta = theta.copy()
    for j in xrange((theta.shape)[1]):
        sum_value = 0.0
        for i in xrange(m):
            sum_value += (tmp_theta*(x[i].getT())-y[0, i])[0,0] * x[i, j]
        theta[0, j] = theta[0, j] - alpha*sum_value/m
    return theta
