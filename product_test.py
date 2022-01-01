import torch
import numpy as np
import torch.nn.functional as F
a = torch.ones(3,4,4)
arr = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
arr1 = np.array([[3,4,1,2,1],[3,4,3,2,1],[1,2,3,4,5]])
b = torch.from_numpy(arr)
b = b.unsqueeze(2)
b = b.unsqueeze(3)
a = a.unsqueeze(1)
print(a)
print(b)
print(a.size())
print(b.size())

product = torch.sum(torch.mul(a,b),dim=1)
print(product.size())
print(product)

