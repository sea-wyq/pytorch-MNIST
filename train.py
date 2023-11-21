import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

class MNIST(torch.nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, 32, 3, 1, 1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(32, 64, 3, 1, 1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(2, 2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 64, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.2),
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 14 * 14 * 64)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if_cuda = torch.cuda.is_available()
    print("if_cuda=",if_cuda)
    gpu_count = torch.cuda.device_count()
    print("gpu_count=",gpu_count)

    model = MNIST().to(device)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)


    epochs = 2
    number = 0
    start =  time.time()
    for epoch in range(epochs):
        # train
        sum_loss = 0.0
        train_correct = 0
        for inputs, lables in train_loader:
            inputs, lables = inputs.to(device),lables.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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