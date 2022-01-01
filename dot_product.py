import torch
import numpy as np
a = torch.ones(2,2,4)
arr = np.array([[[1, 2, 3,4]]])
# 转化为torch的张量形式
b = torch.from_numpy(arr)
print(a)
print(b)
c = torch.mul(a,b)
print(c)
print(c.size())
