'''
线性回归
'''

import torch as t
import matplotlib.pyplot as plt

t.manual_seed(100)
x = t.unsqueeze(t.linspace(-1,1,100),dim=1)# 均分100个-1~1的值 作为100x1的张量
y = 3 * x +2 + 0.2*t.rand(x.size())
plt.figure()
plt.scatter(x.numpy(),y.numpy(), color='blue', marker='o', label='true')
# plt.show()

w = t.randn(1,1,dtype=t.float32, requires_grad=True)
b = t.zeros(1,1,dtype=t.float32, requires_grad=True)
# print(w,b)

lr = 0.001
for i in range(500):
    y_pred = x.mm(w) + b
    loss = 0.5*(y_pred - y)**2
    loss = loss.sum()

    loss.backward()

    with t.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

    if i % 10 == 0:
        print('epoch:%d, w:%.5f, b:%.5f, loss:%.5f'%(i+1,w,b,loss))

# y_pred = x.mm(w) + b
plt.plot(x.numpy(), y_pred.detach().numpy(),'r-', label='predict')
# plt.xlim(-1,1)
# plt.ylim(-1,5)
plt.legend()
plt.show()
