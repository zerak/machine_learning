#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
algorithm 2

COSTFUNCTION Compute cost and gradient for logistic regression
J = COSTFUNCTION(theta, x, y) computes the cost of using theta as the parameter for logistic regression

a. theta is 1*(n+1) vector
   [theta0, theta1, ... thetan]
b. x is m*(n+1) matrix
   set x0 = 1
   [1, x1, ... xn]
   [1, x1, ... xn]
        ......
   [1, x1, ... xn]
   [1, x1, ... xn]
c. y is 1*m vector
   [y1, y2, ... ym]

return real number
"""

import os
import sys
import sigmoid
import numpy as np
from math import log

def cost_function(theta, x, y):
    # number of training examples
    m = (y.shape)[1]
    value = 0.0
    for i in xrange(m):
        sigmoid_value = sigmoid.sigmoid(theta * x[i].getT())
        value += y[0, i]*log(sigmoid_value) + (1-y[0, i])*log(1.0-sigmoid_value)
    return -1.0*value/m
