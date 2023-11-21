import os
import time
from torchvision import datasets, transforms
import torch
import torch.distributed as dist
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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


def save_checkpoint(epoch, model, optimizer, path):
    torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimize_state_dict": optimizer.state_dict(),
}, path)

def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint

def train():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    
    train_dataset = datasets.MNIST(root='./data',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)
    train_sampler = ElasticDistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=100,
                                                num_workers=0,
                                                pin_memory=True,
                                                sampler=train_sampler,)
    
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) train worker starting...")
    model = MNIST().cuda(local_rank)
    ddp_model = DDP(model, [local_rank])
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters())
    max_epoch = 10
    first_epoch = -1
    ckp_path = "checkpoint.pt"
    if os.path.exists(ckp_path):
        print(f"load checkpoint from {ckp_path}")
        checkpoint = load_checkpoint(ckp_path)
        ddp_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimize_state_dict"])
        first_epoch = checkpoint["epoch"]
    for epoch in range(first_epoch + 1, max_epoch):
        # train
        start =  time.time()
        number = 0
        sum_loss = 0.0
        for inputs, lables in train_loader:
            optimizer.zero_grad()
            outputs = ddp_model(inputs.to(local_rank))
            lables = lables.to(local_rank)
            loss = cost(outputs, lables)
            loss.backward()
            optimizer.step()
            sum_loss += loss.data
            end = time.time()
            number += 1
            if number % 100 == 0:
                print('epoch: [%d,%d], step: [%d,%d] loss:%.06f, step time:%.06f' %
                    (epoch + 1, max_epoch, number+1, len(train_loader), sum_loss / len(train_loader),  (end-start)/number))
        if rank == 0:
            save_checkpoint(epoch, ddp_model, optimizer, ckp_path)

def run():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    train()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()

# torchrun \
#     --nnodes=1:2\
#     --nproc_per_node=1\
#     --max_restarts=3\
#     --rdzv_id=1\
#     --rdzv_backend=c10d\
#     --rdzv_endpoint="172.17.0.2:1234"\
#     train.py