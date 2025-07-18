#!/usr/bin/env python3
"""
compare_federated.py

– Chia client theo 3 cách (Dirichlet, ClusterEqual, ClusterNonEqual)
– Ép 3 module a1, a2, a3 dùng đúng cùng 1 client split
– Chạy lần lượt a1.train_fedrl, a2.train_fedrl, a3.train_federated_improved
"""

import random
import numpy as np
import torch
from torchvision import datasets, transforms

import a1
import a2
import a3

# 1) Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark   = False

# 2) Dataset & model constructors
DATASETS = [
    {
        'name': 'MNIST',
        'class': datasets.MNIST,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'model_fn_a1': a1.get_cnn10,
        'model_fn_a3': lambda: a3.SimpleCNN(num_classes=10)
    },
    {
        'name': 'FashionMNIST',
        'class': datasets.FashionMNIST,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ]),
        'model_fn_a1': a1.get_cnn_fashion,
        'model_fn_a3': lambda: a3.SimpleCNN(num_classes=10)
    }
]

# 3) Partition types
PARTITIONS = ['Dirichlet', 'ClusterEq', 'ClusterNonEq']

# 4) Args1 for a1.py
class Args1: pass
args1 = Args1()
args1.alpha         = 0.5
args1.num_clients   = 10
args1.local_epochs  = 3
args1.num_rounds    = 100
args1.batch_size    = 64
args1.lr            = 0.01
args1.momentum      = 0.5
args1.policy_lr     = 1e-4
args1.value_lr      = 1e-3
args1.gamma         = 0.99
args1.tau           = 0.02
args1.buffer_size   = 10000
args1.batch_size_rl = 64
args1.prio_alpha    = 0.6
args1.prio_beta     = 0.4
args1.update_steps  = 10
args1.device        = 'cuda' if torch.cuda.is_available() else 'cpu'

# 5) Args2 for a2.py
class Args2: pass
args2 = Args2()
args2.seed            = 123
args2.data_dir        = './data'
args2.save_dir        = './results_a2'
args2.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args2.num_clients     = 10
args2.num_opt_types   = 3
args2.optimizer_fns   = [
    lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
    lambda p: torch.optim.Adam(p, lr=0.001),
    lambda p: torch.optim.RMSprop(p, lr=0.005, alpha=0.9),
]
args2.batch_size      = 64
args2.local_epochs    = 5
args2.rounds          = 100
args2.policy_hidden   = 128
args2.policy_lr       = 1e-3
args2.buffer_size     = 100
args2.var_lambda      = 0.1
args2.server_lambda   = 1.0
args2.state_dim       = args2.num_clients * 2 + 1
args2.alpha           = 0.5
args2.main_ratio      = 0.6
args2.labels_per_client = 2

# 6) Args3 for a3.py
class Args3: pass
args3 = Args3()
args3.seed             = 42
args3.data_path        = './data'
args3.save_dir         = './results_a3'
args3.device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args3.num_clients      = 10
args3.local_epochs     = 3
args3.batch_size       = 32
args3.rounds           = 100
args3.eval_freq        = 5
args3.lr               = 0.01
args3.alpha            = 0.5
args3.main_ratio       = 0.6
args3.labels_per_client= 2

# 7) Compute split once
def compute_split(partition, train_ds, num_clients, alpha, main_ratio, labels_per_client):
    if partition == 'Dirichlet':
        return a1.pareto_partition(train_ds, num_clients, alpha)
    elif partition == 'ClusterEq':
        return a1.cluster_equal_partition(train_ds, num_clients, main_ratio, labels_per_client)
    elif partition == 'ClusterNonEq':
        return a1.cluster_non_equal_partition(train_ds, num_clients, main_ratio, labels_per_client, alpha)
    else:
        raise ValueError(partition)

# 8) run_a1: patch all partition fns to return the precomputed idxs
def run_a1(cfg, partition):
    print(f"\n>>> [a1] {cfg['name']} – {partition}")
    # load data
    train_ds = cfg['class']('./data', train=True, download=True, transform=cfg['transform'])
    test_ds  = cfg['class']('./data', train=False, download=True, transform=cfg['transform'])
    # compute once
    idxs = compute_split(partition, train_ds,
                         args1.num_clients, args1.alpha,
                         args1.main_ratio if hasattr(args1, 'main_ratio') else None,
                         getattr(args1, 'labels_per_client', None))
    # backup originals
    orig_pe = a1.pareto_partition
    orig_ce = a1.cluster_equal_partition
    orig_cne= a1.cluster_non_equal_partition
    # universal patcher
    def forced(ds, *args, **kwargs):
        return idxs
    # apply patches
    a1.pareto_partition          = forced
    a1.cluster_equal_partition   = forced
    a1.cluster_non_equal_partition = forced
    try:
        a1.train_fedrl(train_ds, test_ds, args1, cfg['model_fn_a1'], f"{cfg['name']}_{partition}_a1")
    finally:
        # restore
        a1.pareto_partition          = orig_pe
        a1.cluster_equal_partition   = orig_ce
        a1.cluster_non_equal_partition = orig_cne

# 9) run_a2: tương tự, patch partition fns
def run_a2(cfg, partition):
    print(f"\n>>> [a2] {cfg['name']} – {partition}")
    # load train_ds chỉ để tính idxs
    train_ds = cfg['class']('./data', train=True, download=True, transform=transforms.ToTensor())
    idxs = compute_split(partition, train_ds,
                         args2.num_clients, args2.alpha,
                         args2.main_ratio, args2.labels_per_client)
    # backup & patch
    orig = {}
    for fn in ['pareto_partition','cluster_equal_partition','cluster_non_equal_partition']:
        orig[fn] = getattr(a2, fn)
        setattr(a2, fn, lambda ds, *args, **kw: idxs)
    args2.partition = {'Dirichlet':'dirichlet','ClusterEq':'CE','ClusterNonEq':'CN'}[partition]
    try:
        a2.train_fedrl(args2, cfg['name'], cfg['class'])
    finally:
        for fn in orig:
            setattr(a2, fn, orig[fn])

# 10) run_a3: code gốc tự lo partition bên trong
def run_a3(cfg, partition):
    print(f"\n>>> [a3] {cfg['name']} – {partition}")
    a3.train_federated_improved(cfg['name'], cfg['class'], args3)

# 11) Orchestrator
def main():
    set_seed(42)
    for partition in PARTITIONS:
        set_seed(42)
        for cfg in DATASETS:
            run_a1(cfg, partition)
            run_a2(cfg, partition)
            run_a3(cfg, partition)

if __name__ == '__main__':
    main()
