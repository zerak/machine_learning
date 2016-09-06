#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Algorithm 1

SIGMOID Compute sigmoid functoon
Instructions: Compute the sigmoid of each value of z (z can be a real number, matrix, vector or scalar).
g(z) = 1/(1+e^-z)

return new matrix or vector
"""
import os
import sys
import numpy as np

def sigmoid(z):
    return round((1 + (np.exp(z*-1))) / 1, 2)

