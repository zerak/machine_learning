"""
Compute cost for linear regression
J = compute_cost(x, y, theta)
x y theta all is numpy matrix
a. x is (n+1)*m matrix
   [x0, x1, ... xn]
   [x0, x1, ... xn]
        ......
   [x0, x1, ... xn]
   [x0, x1, ... xn]
b. y is 1*(n+1) matrix
   [y0, y1, ... yn]
c. theta is 1*(n+1) matrix
   [theta0, theta1, ... thetan]
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import datetime
import numpy as np

def cost_function(x, y, theta):
    # number of training examples
    m = y.shape[1]
    value = 0.0
    for i in xrange(m):
        value = value + pow(((theta*x[i].T).getA()[0][0]-y.getA()[0][i]), 2)
    return value/(2*m)
