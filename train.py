import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Model import MNIST


if __name__ == "__main__":
    device = torch.device('cpu')
    model = MNIST().to(device)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

    epochs = 2
    for epoch in range(epochs):
        # train
        sum_loss = 0.0
        train_correct = 0
        for data in train_loader:
            inputs, lables = data
            inputs, lables = Variable(inputs), Variable(lables)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cost(outputs, lables)
            loss.backward()
            optimizer.step()

            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == lables.data)

        print('[%d,%d] loss:%.03f, correct:%.03f' %
              (epoch + 1, epochs, sum_loss / len(train_loader), 100 * train_correct / len(train_dataset)))

        model.eval()
        test_correct = 0
        for data in test_loader:
            inputs, lables = data
            inputs, lables = Variable(inputs), Variable(lables)
            outputs = model(inputs)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == lables.data)
        print("correct:%.3f%%" % (100 * test_correct / len(test_dataset)))
    torch.save(model.state_dict(), "mnist.pkl")
