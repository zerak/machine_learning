import numpy as np
import normal_equation

x = np.matrix('1 2 3 4 5; 1 10 11 12 13')
y = np.matrix('100 200')
print normal_equation.normal_equation(x, y)
