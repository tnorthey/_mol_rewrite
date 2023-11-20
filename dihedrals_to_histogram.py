import numpy as np

a = np.loadtxt('out')
hist = np.histogram(a, bins=40)
for i in range(len(hist[0])):
    print('%9.8f %i' % (hist[1][i], hist[0][i]))

