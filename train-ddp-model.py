import os
import sys
import tempfile
import torch
import time
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets, transforms

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ModelParallelMNIST(torch.nn.Module):
    def __init__(self,dev0, dev1):
        super(ModelParallelMNIST, self).__init__()
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
        x = x.to(self.dev0)
        x = self.conv(x)
        x = x.view(-1, 14 * 14 * 64)
        x = x.to(self.dev1)
        x = self.dense(x)
        return x

def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    train_dataset = datasets.MNIST(root='./data',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                num_replicas=world_size,
                                                                rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=100,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ModelParallelMNIST(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    cost = torch.nn.CrossEntropyLoss().to(rank)
    optimizer = torch.optim.Adam(ddp_mp_model.parameters())
    
    epochs = 10
    number = 0
    start =  time.time()
    for epoch in range(epochs):
        # train
        sum_loss = 0.0
        train_correct = 0
        for inputs, lables in train_loader:
            optimizer.zero_grad()
            outputs = ddp_mp_model(inputs)
            lables = lables.to(dev1)
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

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus//2
    run_demo(demo_model_parallel, world_size)
    print("finished")

# 单机双卡环境测试完成 