import numpy as np

def hamming(a,b):
    return np.sum(a ^ b) / a.shape[0]

a = np.array([0,1,1,0,1])
b = np.array([1,1,1,0,1])

c = a ^ b
d = np.sum(a^b)

print(a,b,c,d)

h = hamming(a, b)

print(h)
