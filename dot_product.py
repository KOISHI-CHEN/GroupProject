import torch
import numpy as np
import torch.nn.functional as F
a = torch.ones(3,5,4,4)
arr = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
arr1 = np.array([[3,4,1,2,1],[3,4,3,2,1],[1,2,3,4,5]])
b = torch.from_numpy(arr)
b = b.unsqueeze(2)
b = b.unsqueeze(3)

b2 = torch.from_numpy(arr1)
b2 = b2.unsqueeze(2)
b2 = b2.unsqueeze(3)

print(a.size())
print(b.size())
print(a)
print(b)
c = torch.mul(a,b)
s = torch.sum(c,dim=1)
print(c)
print(c.size())

print(s)
print(s.size())

# s = F.softmax(s,0)

print(s)
print(s.size())

s2 = torch.sum(torch.mul(a,b2),dim=1)
print(s2)
print(s2.size())
# torch.Size([2, 256, 78, 78])
# torch.Size([2, 256])
# torch.Size([2, 256])

abc_cat = torch.stack([s,s2],axis = 1)
final = F.softmax(abc_cat,1)
print(abc_cat.size())
print(abc_cat)
print(final.size())
print(final)
print(final[:,1].size())
print(final[:,1])
print(final[:,0])