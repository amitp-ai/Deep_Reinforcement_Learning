
dict1 = {'k1': 1, 'k2': 2}
print(dict1)

print(10*float('inf'))

if not {}: print(1)
else: print(0)

import numpy as np

a = np.array([1,2,3,4,5])
print(a[[False,True,True,False,False]])

a = [1,2,3,4,4,4]
b = (i**2 for i in a)
print(max(b))

import torch

a = [torch.tensor(1),torch.tensor(2),torch.tensor(3)]
print(torch.stack(a, dim=0))
print(float(torch.tensor(1)))

