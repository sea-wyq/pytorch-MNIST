import torch
import torch.nn as nn
import torch.optim as optim
import time 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class PipelineModelParallelMNIST(torch.nn.Module):
    def __init__(self,dev0, dev1,split_size=10):
        super(PipelineModelParallelMNIST, self).__init__()
        self.split_size = split_size
        self.dev0 = dev0
        self.dev1 = dev1
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, 32, 3, 1, 1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(32, 64, 3, 1, 1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(2, 2)).to(dev0)
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 64, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.2),
                                         torch.nn.Linear(1024, 10)).to(dev1)

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        # calc on cuda:0, copy output to cuda:1
        s_prev = self.conv(s_next).to(self.dev1)
        s_prev = s_prev.view(-1, 14 * 14 * 64)
        ret = []

        for s_next in splits:
            s_prev = self.dense(s_prev)
            ret.append(s_prev.view(s_prev.size(0), -1))
            s_prev = self.conv(s_next).to(self.dev1)
            s_prev = s_prev.view(-1, 14 * 14 * 64)

        s_prev = self.dense(s_prev)
        ret.append(s_prev.view(s_prev.size(0), -1))

        return torch.cat(ret)


def train():

    train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    
    dev0 = 0
    dev1 = 1
    model = PipelineModelParallelMNIST(dev0,dev1)
    model.train(True)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 10
    number = 0
    start =  time.time()
    for epoch in range(epochs):
        # train
        sum_loss = 0.0
        train_correct = 0
        for inputs, lables in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(dev0))
            lables = lables.to(outputs.device)
            loss = cost(outputs, lables)
            loss.backward()
            optimizer.step()
            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == lables.data)
            end = time.time()
            number += 1
        print('[%d,%d] loss:%.03f, correct:%.03f' %
              (epoch + 1, epochs, sum_loss / len(train_loader), 100 * train_correct / len(train_dataset)))
        print("train time: ",(end-start)/number )
    torch.cuda.synchronize()


if __name__=="__main__":
    train()

# python train.py 单机双卡环境测试完成