import numpy as np
import matplotlib.pyplot as plt

n1 = 100
n2 = 100
mu1 = 5
sig1 = 10
mu2 = 10
sig2 = 10

a = np.random.normal(loc = mu1, scale = sig1)
b = np.random.normal(loc = mu2, scale = sig2)

c = np.hstack((a,b))

plt.hist(c)
plt.show()