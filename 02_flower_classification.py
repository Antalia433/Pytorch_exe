'''
鸢尾花分类
'''
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import sklearn.model_selection as ms
from torch.utils.data import TensorDataset, DataLoader

x_data = load_iris().data
y_data = load_iris().target
print(x_data.shape, y_data.shape)  # (150, 4) (150,)

train_x, test_x, train_y, test_y = ms.train_test_split(x_data, y_data, test_size=0.2, random_state=115, shuffle=True)

train_x = torch.from_numpy(train_x).type(torch.float32)
train_y = torch.from_numpy(train_y).type(torch.int64)
test_x = torch.from_numpy(test_x).type(torch.float32)
test_y = torch.from_numpy(test_y).type(torch.LongTensor)

batch_size = 32
epochs = 300

train_data_0 = TensorDataset(train_x, train_y)
train_data = DataLoader(train_data_0, batch_size=batch_size)
test_data_0 = TensorDataset(test_x, test_y)
test_data = DataLoader(test_data_0, batch_size=batch_size)

w = torch.normal(mean=0, std=0.01, size=[4, 3], dtype=torch.float32, requires_grad=True)  # 四种特征 三种类别
b = torch.normal(mean=0, std=0.01, size=[3], dtype=torch.float32, requires_grad=True)
# print(w, b)

lr = 0.1
train_loss = []
test_acc = []
loss_all = 0
loss_func = nn.CrossEntropyLoss()

for epoch in range(epochs):
    s = 0
    for x_train, y_train in train_data:
        s += 1
        y = torch.matmul(x_train, w) + b
        # print(y.shape) #torch.Size([32, 3])
        y = nn.functional.softmax(y)

        # y_true = nn.functional.one_hot(y_train, 3)
        # loss = torch.mean(torch.pow(y_true - y, 2))  ########
        loss = loss_func(y,y_train)
        loss_all += loss.detach().numpy()
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

            w.grad.zero_()
            b.grad.zero_()

    train_loss.append(loss_all / s)
    print('epoch:%d, loss:%f' % (epoch, train_loss[epoch]))
    loss_all = 0

    # test

    total_number = 0
    total_correct = 0
    for x_test, y_test in test_data:
        y = torch.matmul(x_test, w) + b
        y = nn.functional.softmax(y)
        # print(y)  # 32x3
        pred = torch.argmax(y, axis=1).type(y_test.type())
        # print(pred,y_test)    # [32],[32]
        correct = torch.eq(pred,y_test).type(torch.int32)# eq对比T/F  并转为整形
        correct = torch.sum(correct)

        total_correct += int(correct)
        total_number += x_test.shape[0]

    acc = total_correct/total_number
    test_acc.append(acc)
    print('Test_acc: %.4f'%(acc))

plt.subplot(2,1,1)
plt.xlabel(' ')
plt.ylabel('loss')
plt.plot(train_loss, label='loss')

plt.subplot(2,1,2)
plt.xlabel('epoch')
plt.ylabel('test_acc')
plt.plot(test_acc, label='test_acc')
plt.legend()
plt.show()


