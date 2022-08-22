# -*- coding: utf-8 -*-
"""
Created on 2022/8/8 下午5:58

@Author: chenjun

@Describe:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1
--master_addr="172.31.3.103" --master_port=23457 demo2.py
"""
import torch
import socket
import argparse
import os
import random
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
# 是否启用SyncBatchNorm
parser.add_argument('--syncBN', type=bool, default=True)
parser.add_argument('--dist-backend', default = 'nccl', type = str,
                    help = 'distributed backend')
parser.add_argument('--seed', type = int, default = 40, help = "train epoch")

args = parser.parse_args()


# RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
params = {k: v for k, v in args.__dict__.items()
          for i in ['RANK', 'WORLD', 'MASTER', 'GROUP'] if i in k}
if params:
    print(params)


class MyModel(nn.Module):
    def __init__(self) -> None:
        super(MyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def check_dist():
    needed = ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
    # print(os.environ.__dict__)
    # print([os.getenv(col) for col in needed])
    if not all([os.getenv(col) for col in needed]) or not torch.cuda.is_available():
        args.distributed = False
        return
    local_size = int(os.getenv('LOCAL_WORLD_SIZE', 1))
    local_count = torch.cuda.device_count()
    if local_size > local_count:
        raise ValueError(f'LOCAL_WORLD_SIZE `{local_size}` is beyond '
                         f'local node GPU count `{local_count}`')
    args.distributed = True
    return


def task_worker(args):
    check_dist()

    if args.distributed:
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv('LOCAL_RANK'), 0)
        rank = int(os.getenv('RANK'), 0)

        set_random_seed(args.seed)
        dist.init_process_group(backend = args.dist_backend, init_method = 'env://')
        # dist.init_process_group(backend = args.dist_backend, init_method = 'env://',
        #                         rank = rank, world_size = world_size)

        print(f'global_rank = {rank} local_rank = {local_rank}'
              f'world_size = {world_size}, hostname = {socket.gethostname()}')
    else:
        if torch.cuda.is_available():
            print('use single process, gpu `0`...')
            local_rank = 0
        else:
            print('use single process and cpu...')
            local_rank = None

    run_model(local_rank, args)


def run_model(local_rank, args):
    device = torch.device(local_rank) if local_rank is not None else torch.device('cpu')

    model = MyModel().to(device)
    if args.distributed:
        print(f'model params in {local_rank} before ddp:', list(model.parameters())[0][0, 0])
        model = DDP(model, device_ids = [local_rank])
        print(f'model params in {local_rank} after ddp:', list(model.parameters())[0][0, 0])

    loss_fn = nn.MSELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr = 0.001)

    optimizer.zero_grad()
    outputs = model(torch.randn(20, 10).to(device))
    labels = torch.randn(20, 5).to(device)
    loss = loss_fn(outputs, labels)
    print(f"get basic loss `{loss.item()}` on rank {local_rank}.")
    loss.backward()
    optimizer.step()
    print(f'updated model params in {local_rank}:', list(model.parameters())[0][0, 0])

    # wait until all processed completed, not necessary
    # if device != torch.device("cpu") and args.distributed:
    #     torch.cuda.synchronize(device)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    task_worker(args)
