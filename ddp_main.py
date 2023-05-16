################
## main.py文件
import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import math
import random
import numpy as np
import torch.backends.cudnn as cudnn

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

### 1. 基础模块 ### 
# 假设我们的模型是这个，与DDP无关
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 来源：https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

# 合并结果的函数
# 1. all_gather，将各个进程中的同一份数据合并到一起。
#   和all_reduce不同的是，all_reduce是平均，而这里是合并。
# 2. 要注意的是，函数的最后会裁剪掉后面额外长度的部分，这是之前的SequentialDistributedSampler添加的。
# 3. 这个函数要求，输入tensor在各个进程中的大小是一模一样的。
def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

# 假设我们的数据是这个
def get_dataset(root='./data', batch_size=1):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    my_trainset = torchvision.datasets.CIFAR10(root=root, train=True, 
        download=True, transform=transform)
    my_testset = torchvision.datasets.CIFAR10(root=root, train=False, 
        download=True, transform=transform)
    # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
    #      用，就完事儿！sampler的原理，第二篇中有介绍。
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。
    #      也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(my_trainset, 
        batch_size=batch_size, num_workers=4, sampler=train_sampler)
    
    # 使用我们的新sampler
    test_sampler = SequentialDistributedSampler(my_testset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(my_testset, batch_size=batch_size, sampler=test_sampler)

    return trainloader, testloader

def my_evaluate_func(predictions, labels):
    pass
    
### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--epoch", default=10, type=int)
parser.add_argument("--load_from", default=None, type=str)
parser.add_argument("--save_to", default='./ckpt/', type=str)

args = parser.parse_args()
local_rank = args.local_rank

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# local_rank = dist.get_rank()
print('local rank: ',dist.get_rank())
init_seeds(local_rank+1)

# 准备数据，要在DDP初始化之后进行
trainloader, testloader = get_dataset(root='./data',batch_size=args.batch_size)

# 构造模型
model = ToyModel().to(local_rank)
# DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = args.load_from
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))
# DDP: 构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 假设我们的loss是这个
loss_func = nn.CrossEntropyLoss().to(local_rank)

### 3. 网络训练  ###
model.train()
iterator = tqdm(range(args.epoch))
for epoch in iterator:
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in trainloader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        optimizer.step()
    # DDP:
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), args.save_to+"%d.ckpt" % epoch)
    
    with torch.no_grad():
        # 1. 得到本进程的prediction
        predictions = []
        labels = []
        for data, label in testloader:
            data, label = data.to(local_rank), label.to(local_rank)
            predictions.append(model(data))
            labels.append(label)
        # 进行gather
        predictions = distributed_concat(torch.concat(predictions, dim=0), 
                                         len(testloader.sampler.dataset))
        labels = distributed_concat(torch.concat(labels, dim=0), 
                                    len(testloader.sampler.dataset))
        # 3. 现在我们已经拿到所有数据的predictioin结果，进行evaluate！
        my_evaluate_func(predictions, labels)


################
## Bash运行
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py