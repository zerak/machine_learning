"""
Compute cost for linear regression
J = compute_cost(x, y, theta)
x y theta all is numpy matrix
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

return real number
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

def cost_function(x, y, theta):
    # number of training examples
    m = (y.shape)[1]
    value = 0.0
    for i in xrange(m):
        value += pow((theta*(x[i].getT())-y[0, i])[0, 0], 2)
    return value/(2*m)

