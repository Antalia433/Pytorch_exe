'''
线性回归/sigmod回归
'''
import math

import torch as t
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

t.manual_seed(100)
x = t.unsqueeze(t.linspace(-10, 10, 100), dim=1)  # 均分100个-1~1的值 作为100x1的张量
# print(x.shape[0])
# x = x.permute(1, 0)
# print(x.numpy()[2])
# assert 0
y = 1 / (1 + t.exp(-x))
# print(y)
plt.figure()
plt.scatter(x.numpy(), y.numpy(), color='blue', marker='o', label='true')
# plt.show()
model = nn.Sequential(
    nn.Linear(1, 1),
    # nn.ReLU(),
    nn.Sigmoid(),
    # nn.Linear(10,1)
)

w = t.randn(1, 1, dtype=t.float32, requires_grad=True)
b = t.zeros(1, 1, dtype=t.float32, requires_grad=True)
# print(w,b)

lr = 0.01
loss_fun = nn.MSELoss()
loss_all = []
opti = t.optim.Adam(model.parameters(), lr=lr)
for i in range(100):
    # y_pred = x.mm(w) + b
    y_pred = model(x)
    # print(x.shape,y_pred.shape)

    loss = loss_fun(y_pred, y)

    opti.zero_grad()
    loss.backward()
    opti.step()

    # with t.no_grad():
    #     w -= lr * w.grad
    #     b -= lr * b.grad
    #
    #     w.grad.zero_()
    #     b.grad.zero_()

    if i % 10 == 0:
        # print('epoch:%d, w:%.5f, b:%.5f, loss:%.5f'%(i+1,w,b,loss))
        print('epoch:%d, loss:%.5f' % (i + 1, loss.item()))

# print(x, y_pred)
t_x = t.tensor([0.5])
print(model(t_x))
# y_pred = x.mm(w) + b
# plt.scatter(1,y_pred.item())
# plt.plot([x.numpy()[0][i] for i in range(x.shape[0])],
#          [y_pred.detach().numpy()[0][i] for i in range(y.shape[0])], 'r-', label='predict')
plt.plot(x.numpy(), y_pred.detach().numpy(), 'r-', label='predict')
# plt.xlim(-1,1)
# plt.ylim(-1,5)
plt.legend()
plt.show()
