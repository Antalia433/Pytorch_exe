import torch.utils.data as Data
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from PIL import Image


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

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        # out = F.softmax(out)

        return out


def calculate_accuracy(pred, y):
    #    print(pred)  #batch_sizex10
    pred = pred.cpu()
    y = y.cpu()
    pred = torch.max(pred, 1)[1].data.numpy()
    #    print(pred) #1x16
    acc = float((pred == y.data.numpy()).astype(int).sum()) / float(y.size(0))
    return acc


def for_train():
    epochs = 3
    batch_size = 16
    lr = 0.001

    train_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize([224, 224])
    ])

    train_data = ImageFolder('flower/', transform=train_transformations)

    # print(train_data.class_to_idx) #{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}

    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True)

    model = CNN(3, 5)
    summary(model, (3, 28, 28))
    # assert 0

    cuda_avail = torch.cuda.is_available()

    if cuda_avail:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    for e in range(epochs):
        e_loss = []
        e_acc = []
        model.train()
        for train_x, train_y in train_loader:
            if cuda_avail:
                train_x = train_x.cuda()
                train_y = train_y.cuda()

            pred = model(train_x)
            # print(pred, train_y)
            # assert 0
            loss = loss_func(pred, train_y)

            e_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = calculate_accuracy(pred, train_y)
            e_acc.append(acc)

        epoch_loss = sum(e_loss) / len(e_loss)
        epoch_acc = sum(e_acc) / len(e_acc)
        print(f'train epoch: {e}, loss: {epoch_loss}, acc:{epoch_acc}')

    torch.save(model, 'f_cnn.pkl')
    print('finish')


def for_test(resume_path):
    class_dict = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}

    test_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize([112, 112])
    ])
    img_p = 'test_img/1.png'
    img = Image.open(img_p)
    img = test_transformations(img)
    print(img.shape)
    img.unsqueeze_(dim=0)
    print(img.shape)

    # model = CNN(3,5)
    resume_path = resume_path
    print('读取已训练模型 {}'.format(str(resume_path)))
    model = torch.load(resume_path, map_location=torch.device('cpu'))

    # model.load_state_dict(checkpoint['model'])
    # epoch = checkpoint['epoch']
    pred = model(img)
    p = F.softmax(pred, 1)
    # print(p, pred.shape)
    # _, ind = torch.max(pred.data, 1)  # 不.data会带梯度    tensor([0.0085]) tensor([1])
    _, ind = torch.max(p.data, 1)  # 不.data会带梯度    tensor([0.2107]) tensor([1])
    # print(_, ind)

    for i, k in class_dict.items():
        if ind.item() == k:
            print('预测结果为:' + i)


if __name__ == '__main__':
    resume_path = 'f_cnn.pkl'
    for_test(resume_path)
