import torch


a = torch.ones(12, 16)
x1 = a.unsqueeze(1)
print(x1.size())
print((2, 16) + a.shape[:1])
x2 = a.unsqueeze(1).expand(a.shape[:1] + (2, 16)).contiguous().view(a.shape[:1] (16 * 2,) + a.shape[1:])
b = a.unsqueeze(1) # (16, 1, 12, 512)
c = a.shape[1:]
d = (16, 2) + a.shape[1:]
e = (16 * 2,) + a.shape[1:]


print(x2.size())
#x2 = x2.unsqueeze(1).expand((bs, sample) + x2.shape[1:]).contiguous().view((bs * sample,) + x2.shape[1:])