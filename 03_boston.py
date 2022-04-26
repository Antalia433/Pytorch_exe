'''
波士顿房价预测
'''
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

data = load_boston()
x = data['data']
data_y = data['target']
y = data_y.reshape(-1, 1)
# print(x.shape,data_y.shape,y.shape)  #(506, 13) (506,) (506, 1)

x = MinMaxScaler().fit_transform(x)  # 数据归一化

x = torch.from_numpy(x).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.FloatTensor)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=48)

model = nn.Sequential(
    nn.Linear(13, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

criterion = nn.MSELoss() #均方差
optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
epochs = 500
loss_all = []
for i in range(epochs):
    pred = model(train_x)
    loss = criterion(pred, train_y)
    loss_all.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('over loss:',loss_all[-1])
# 测试
output = model(test_x)
predict_list = output.detach().numpy()

plt.subplot(2, 1, 1)
plt.plot(loss_all, label='train_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.tight_layout(h_pad=2)

plt.subplot(2, 1, 2)
plt.scatter(range(test_x.shape[0]), test_y, label='input', color='blue')
plt.scatter(range(test_x.shape[0]), predict_list, label='result', color='tomato')
plt.legend()
plt.xlabel('test_id')
plt.ylabel('cost')
plt.tight_layout(h_pad=2)
plt.show()
