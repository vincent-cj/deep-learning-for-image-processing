# -*- coding: utf-8 -*-
"""
Created on 2022/8/10 下午4:45

@Author: chenjun

@Describe:
CUDA_VISIBLE_DEVICES=0 python demo3.py --dist-url 'tcp://172.31.3.103:2589' 
--dist-backend 'nccl' --world-size 2 --rank 0 --nproc-per-node 2

利用mp.spawn进行启动时，无法对每个节点分别传入不同的 `nproc_per_node` 参数,
来影响全局 `world_size`，进而对每个节点分别调用不同的GPU数量用以训练
如果想实现每个节点调用不同数量的GPU，那么需要提前计算好全局GPU数量，
并赋值到 `world_size` 中，并在不同节点的代码中更改 `rank` 的值
"""

import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:12346', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--nproc-per-node', default=1, type=int,
                    help='num gpu used in current node')
parser.add_argument('--distributed', action='store_true',
                    help='if or not distributed')
args = parser.parse_args()


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
    if not torch.cuda.is_available():
        args.distributed = False
        args.locl_rank = None
        print('single process, use `cpu`...')
        return
    local_size = args.nproc_per_node
    local_count = torch.cuda.device_count()
    if local_size > local_count:
        raise ValueError(f'LOCAL_WORLD_SIZE `{local_size}` is beyond '
                         f'local node GPU count `{local_count}`')
    return


def launch_worker():
    check_dist()
    if args.distributed:
        print('distributed processes...')
        nproc_per_node = args.nproc_per_node
        mp.spawn(single_process, nprocs = nproc_per_node, args = (args,))
    else:
        if torch.cuda.is_available():
            print('use single process, gpu `0`...')
            local_rank = 0
        else:
            print('use single process and cpu...')
            local_rank = None
        single_process(local_rank, args)


def single_process(local_rank, args):

    if args.distributed:
        print("Use GPU: {} for training".format(local_rank))

        # args.world_size should be assigned manully according global num_proc before launch script.py
        # args.rank = args.rank * args.nproc_per_node + gpu     # num_proc for every node is equal
        # args.rank = 1 + local_rank   # num_proc for node1 is 1 and for node2 is multiple
        print('got args: ', args.__dict__)

        # init_method 默认为 env:// 时，需要在环境变量配置主进程的地址和端口，
        # 或者在dist.init_process_group 传入主进程的地址和端口
        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = "12345"
        # dist.init_process_group(backend = args.dist_backend,
        #                         world_size = args.world_size, rank = args.rank)

        # 手工传入
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        global_rank = dist.get_rank()
        print(f'global_rank = {global_rank} local_rank = {local_rank}'
              f'world_size = {args.world_size}')
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    launch_worker()

