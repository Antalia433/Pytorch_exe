'''
手写体分类
'''

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, input_channels, classes):
        super(CNN, self).__init__()
        self.conv1 = self._make_layers(input_channels, 16)
        self.conv2 = self._make_layers(16, 32)
        self.fc1 = nn.Linear(32 * 7 * 7, classes)

    def _make_layers(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = F.softmax(out)

        return out

def calculate_accuracy(pred, y):
#    print(pred)  #batch_sizex10
    pred = pred.cpu()
    y = y.cpu()
    pred = torch.max(pred, 1)[1].data.numpy()
#    print(pred) #1x16
    acc = float((pred == y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    return acc


epochs = 3
batch_size = 16
lr = 0.001

train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)
test_data = torchvision.datasets.MNIST(
        root='./data', train=False,
        transform=torchvision.transforms.ToTensor(), download=False)

print(train_data.train_data.size(),
      test_data.test_data.size())  # torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])

# print(train_data.train_labels.size()) #torch.Size([60000])
# plt.imshow(train_data.train_data[44].numpy(), cmap='Greys')
# plt.show()

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=batch_size,
                               shuffle=True)
test_loader = Data.DataLoader(dataset=test_data,
                              batch_size=batch_size,
                              shuffle=False)

model = CNN(1,10)

cuda_avail = torch.cuda.is_available()

if cuda_avail:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

for e in range(epochs):
    e_loss = []
    e_acc = []
    model.train()
    for train_x,train_y in train_loader:
        if cuda_avail:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        pred = model(train_x)
        # print(pred, train_y)
        # assert 0
        loss = loss_func(pred,train_y)

        e_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(pred,train_y)
        e_acc.append(acc)

    epoch_loss = sum(e_loss) / len(e_loss)
    epoch_acc = sum(e_acc) / len(e_acc)
    print(f'train epoch: {e}, loss: {epoch_loss}, acc:{epoch_acc}')

    model.eval()

    t_loss = []
    t_acc = []
    for test_img,test_y in test_loader:
        if cuda_avail:
            test_img = test_img.cuda()
            test_y = test_y.cuda()

        pred = model(test_img)
        loss = loss_func(pred,test_y)

        t_loss.append(loss.item())
        t_acc.append(calculate_accuracy(pred,test_y))

    epoch_loss = sum(t_loss) / len(t_loss)
    epoch_acc = sum(t_acc) / len(t_acc)
    print(f'test epoch: {e}, loss: {epoch_loss}, acc:{epoch_acc}')


torch.save(model,'cnn.pkl')
print('finish')


