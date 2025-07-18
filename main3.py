#!/usr/bin/env python3
"""
federated_compare.py - VERSION WITH METRICS LOGGING

Standalone script implementing three federated learning algorithms
FedAvg
FedRL with optimizer-selection DRL
FedRL actor-critic aggregation

Now includes logging of metrics (accuracy, reward) per round,
saving to CSV and plotting to PNG.
"""

import os
import copy
import pickle  # để lưu/ load nhanh với cấu trúc Python
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from copy import deepcopy

# -------------------- Utils --------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ép PyTorch dùng deterministic algorithms, tắt benchmark để đảm bảo kết quả tái lập
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ClientDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, indices):
        self.base = base_ds      # Dataset gốc (toàn bộ)
        self.idxs = list(indices)  # Danh sách chỉ số mẫu cho client này

    def __len__(self):
        return len(self.idxs)    # Kích thước dataset của client

    def __getitem__(self, i):
        # Lấy i-th sample của client, ánh xạ chỉ số local sang chỉ số trong dataset gốc
        return self.base[self.idxs[i]]

def evaluate(model, loader, device):
    model.eval()                # Chuyển sang chế độ inference
    correct = total = 0
    with torch.no_grad():       # Tắt tính toán gradient để tiết kiệm bộ nhớ
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)  # Lấy nhãn dự đoán
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total      # Trả về accuracy

# -------------------- Partitions --------------------


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users

# -------------------- Model --------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 1 kênh đầu vào (grayscale), 32 filter, kernel 3x3, stride=1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 32->64 filter, kernel 3x3, stride=1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Pooling 2x2 để giảm kích thước ảnh
        self.pool = nn.MaxPool2d(2)
        # Fully connected: flatten đầu ra 64*12*12 -> 128
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        # Fully connected: 128 -> num_classes
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv -> ReLU
        x = F.relu(self.conv1(x))
        # Conv -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten từ kích thước (batch,64,12,12) về (batch,64*12*12)
        x = torch.flatten(x, 1)
        # FC -> ReLU
        x = F.relu(self.fc1(x))
        # FC cuối cùng trả logits
        return self.fc2(x)

# -------------------- Algorithms --------------------
# 1) FedAvg
class FedAvgServer:
    def __init__(self, model, device):
        # Khởi tạo global model và chuyển lên device (CPU hoặc GPU)
        self.global_model = model.to(device)
        self.device = device

    def aggregate(self, client_models, weights):
        # weights là danh sách kích thước mỗi client (số mẫu)
        gdict = self.global_model.state_dict()
        # reset toàn bộ grads/nguyên liệu về 0
        for k in gdict:
            gdict[k] = torch.zeros_like(gdict[k])
        total = sum(weights)
        # Tổng hợp state_dict của client theo trọng số
        for cm, w in zip(client_models, weights):
            sd = cm.state_dict()
            for k, v in sd.items():
                # cộng dồn v*(w/total)
                gdict[k] += v.to(self.device) * (w / total)
        # Load lại vào global model
        self.global_model.load_state_dict(gdict)

    def aggregate2(self, client_models, weights=None):
        # weights không còn dùng nữa, chỉ cần số clients
        n = len(client_models)
        # Lấy state_dict của client đầu làm khung
        gdict = copy.deepcopy(client_models[0].state_dict())
        # Khởi tạo zeros
        for k in gdict:
            gdict[k] = torch.zeros_like(gdict[k])

        # Cộng dồn đồng đều
        for cm in client_models:
            sd = cm.state_dict()
            for k, v in sd.items():
                gdict[k] += v.to(self.device) / n

        # Load về global model
        self.global_model.load_state_dict(gdict)

def train_fedavg(name, dataset, args, split_idxs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = args.device
    loaders, sizes = [], []
    # 1) Chuẩn bị DataLoader cho từng client từ split_idxs
    for idxs in split_idxs:
        ds = ClientDataset(dataset['train'], idxs)
        sizes.append(len(ds))
        loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=True))
    # DataLoader cho test chung
    test_loader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False)

    # Khởi tạo server với model SimpleCNN
    server = FedAvgServer(SimpleCNN(), device)
    history = {'round': [], 'accuracy': []}

    # 2) Vòng lặp rounds
    for r in range(1, args.rounds + 1):
        client_models = []
        # Mỗi client: copy global model, train local và trả về model
        for loader in loaders:
            local_model = deepcopy(server.global_model).to(device)
            opt = optim.SGD(local_model.parameters(), lr=args.lr, momentum=0.9)
            local_model.train()
            for _ in range(args.local_epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = F.cross_entropy(local_model(x), y)
                    loss.backward()
                    opt.step()
            client_models.append(local_model)

        # 3) Aggregate lên server
        #server.aggregate(client_models, sizes)
        server.aggregate2(client_models)

        # 4) Đánh giá định kỳ
        if r % args.eval_freq == 0 or r == args.rounds:
            acc = evaluate(server.global_model, test_loader, device)
            print(f"[FedAvg][{name}][{r}/{args.rounds}] Test Acc={acc*100:.2f}%")
            history['round'].append(r)
            history['accuracy'].append(acc)

    # 5) Lưu model và metrics
    torch.save(server.global_model.state_dict(), os.path.join(out_dir, 'model.pth'))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    # Vẽ đồ thị accuracy qua các round

    plt.figure()
    plt.plot(history['round'], history['accuracy'])
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy')
    plt.title(f'FedAvg on {name}')
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    plt.close()

# 2) FedRL-DRL (Dùng reinforcement learning chọn hàm tối ưu)
class DRLServer:
    def __init__(self, model_fn, args, sizes):
        self.device = args.device
        # Đưa sizes (số mẫu mỗi client) về dạng tensor
        self.data_fracs = torch.tensor(sizes, dtype=torch.float, device=self.device)
        # Global model khởi tạo từ function trả về model mới
        self.global_model = model_fn().to(self.device)
        # Chính sách (policy) output len(sizes)*num_opt_types logits
        self.policy = nn.Sequential(
            nn.Linear(len(sizes) + 1, 128),  # +1 cho reward trước
            nn.LeakyReLU(),
            nn.Linear(128, len(sizes) * args.num_opt_types)
        ).to(self.device)
        self.opt = optim.Adam(self.policy.parameters(), lr=args.policy_lr)
        self.num_clients = len(sizes)
        self.num_opt = args.num_opt_types

    def select(self, state):
        # state: [data_fracs_normalized..., prev_accuracy]
        logits = self.policy(state)
        # Softmax và reshape thành (num_clients, num_opt)
        probs = F.softmax(logits, dim=-1).view(self.num_clients, self.num_opt)
        dist = Categorical(probs)
        acts = dist.sample()            # chọn tối ưu type cho mỗi client
        return acts.cpu().tolist(), dist.log_prob(acts).sum()

    def aggregate(self, client_models):
        # Tương tự FedAvg, nhưng weights dùng self.data_fracs
        gdict = self.global_model.state_dict()
        for k in gdict: gdict[k] = torch.zeros_like(gdict[k])
        total = self.data_fracs.sum().item()
        for cm, w in zip(client_models, self.data_fracs.tolist()):
            sd = cm.state_dict()
            for k, v in sd.items():
                gdict[k] += v.to(self.device) * (w / total)
        self.global_model.load_state_dict(gdict)


def train_fedrl_drl(name, dataset, args, split_idxs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = args.device
    loaders, sizes = [], []
    for idxs in split_idxs:
        ds = ClientDataset(dataset['train'], idxs)
        sizes.append(len(ds))
        loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=True))
    test_loader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False)

    server = DRLServer(lambda: SimpleCNN(), args, sizes)
    # baseline: accuracy ban đầu
    baseline = evaluate(server.global_model, test_loader, device)
    prev = baseline
    history = {'round': [], 'accuracy': [], 'reward': []}

    for r in range(1, args.rounds + 1):
        # Build state gồm data_fracs normal hóa và acc trước
        st = torch.cat([server.data_fracs / server.data_fracs.sum(), torch.tensor([prev], device=device)])
        acts, logp = server.select(st.unsqueeze(0))
        client_models = []
        # Local cập nhật với optimizer được chọn bởi DRL
        for i, loader in enumerate(loaders):
            lm = deepcopy(server.global_model).to(device)
            opt_fn = args.optimizer_fns[acts[i]]
            opt = opt_fn(lm.parameters())
            lm.train()
            for _ in range(args.local_epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    F.cross_entropy(lm(x), y).backward()
                    opt.step()
            client_models.append(lm)
        # Aggregate và đánh giá
        server.aggregate(client_models)
        acc = evaluate(server.global_model, test_loader, device)
        reward = (acc - prev) + 0.1 * (acc - baseline)
        # Cập nhật policy
        server.opt.zero_grad()
        (-reward * logp).backward()
        torch.nn.utils.clip_grad_norm_(server.policy.parameters(), 1.0)
        server.opt.step()
        prev = acc

        if r % args.eval_freq == 0 or r == args.rounds:
            print(f"[FedRL-DRL][{name}][{r}/{args.rounds}] Test Acc={acc*100:.2f}%, Reward={reward:.4f}")
            history['round'].append(r)
            history['accuracy'].append(acc)
            history['reward'].append(reward)

    # Lưu kết quả
    torch.save(server.global_model.state_dict(), os.path.join(out_dir, 'model.pth'))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    # Vẽ đồ thị accuracy và reward

    plt.figure()
    plt.plot(history['round'], history['accuracy'])
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title(f'FedRL-DRL on {name}')
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    plt.close()
    plt.figure()
    plt.plot(history['round'], history['reward'])
    plt.xlabel('Round')
    plt.ylabel('Reward')
    plt.title(f'FedRL-DRL Reward on {name}')
    plt.savefig(os.path.join(out_dir, 'reward.png'))
    plt.close()

# 3) FedRL-AC (Actor-Critic)
class ACServer:
    def __init__(self, model_fn, args, sizes):
        self.device = args.device
        self.global_model = model_fn().to(self.device)
        self.data_fracs = torch.tensor(sizes, dtype=torch.float, device=self.device)
        dim = len(sizes) * 2  # state: [loss_i_norm..., frac_i...]
        # Actor network output ra xác suất weight mỗi client
        self.actor = nn.Sequential(
            nn.Linear(dim, 128), nn.LeakyReLU(), nn.Linear(128, len(sizes))
        ).to(self.device)
        # Critic network output giá trị state-value
        self.critic = nn.Sequential(
            nn.Linear(dim, 128), nn.LeakyReLU(), nn.Linear(128, 1)
        ).to(self.device)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=args.policy_lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=args.value_lr)

    def select_weights(self, state):
        # Trả về vector weights sau khi softmax
        return F.softmax(self.actor(state), dim=-1).squeeze()

    def aggregate(self, client_models, weights):
        # Aggregate tương tự FedAvg nhưng dùng weights do actor chọn
        gdict = self.global_model.state_dict()
        for k in gdict: gdict[k] = torch.zeros_like(gdict[k])
        w_norm = weights / weights.sum()
        for cm, w in zip(client_models, w_norm.tolist()):
            sd = cm.state_dict()
            for k, v in sd.items():
                gdict[k] += v.to(self.device) * w
        self.global_model.load_state_dict(gdict)


def train_fedrl_ac(name, dataset, args, split_idxs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = args.device
    loaders, sizes = [], []
    for idxs in split_idxs:
        ds = ClientDataset(dataset['train'], idxs)
        sizes.append(len(ds))
        loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=True))
    test_loader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False)

    server = ACServer(lambda: SimpleCNN(), args, sizes)
    history = {'round': [], 'accuracy': [], 'reward': []}

    for r in range(1, args.rounds+1):
        client_models, losses = [], []
        # 1) Local train và tính loss trung bình mỗi client
        for loader in loaders:
            lm = deepcopy(server.global_model).to(device)
            opt = optim.SGD(lm.parameters(), lr=args.lr, momentum=0.9)
            lm.train()
            total_loss, cnt = 0, 0
            for _ in range(args.local_epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = F.cross_entropy(lm(x), y)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item() * y.size(0)
                    cnt += y.size(0)
            losses.append(total_loss / cnt if cnt>0 else 0)
            client_models.append(lm)
        # 2) Chuẩn hóa loss và frac, tạo state
        norm_loss = [(l - min(losses)) / (max(losses) - min(losses) + 1e-8) for l in losses]
        norm_sz = [s / sum(sizes) for s in sizes]
        state = torch.tensor(norm_loss + norm_sz, dtype=torch.float, device=device).unsqueeze(0)
        # 3) Actor chọn weights
        weights = server.select_weights(state)
        # 4) Aggregate model
        server.aggregate(client_models, weights)
        # 5) Tính reward = -mean(losses)
        reward = -sum(losses) / len(losses)
        # 6) Cập nhật critic
        val = server.critic(state)
        tgt = torch.tensor([[reward]], device=device)
        loss_crit = F.mse_loss(val, tgt)
        server.opt_critic.zero_grad()
        loss_crit.backward()
        torch.nn.utils.clip_grad_norm_(server.critic.parameters(), 1.0)
        server.opt_critic.step()
        # 7) Cập nhật actor với advantage
        adv = reward - val.detach().item()
        logp = torch.log(weights + 1e-8)
        loss_act = -adv * logp.sum()
        server.opt_actor.zero_grad()
        loss_act.backward()
        torch.nn.utils.clip_grad_norm_(server.actor.parameters(), 1.0)
        server.opt_actor.step()
        # 8) Đánh giá định kỳ
        if r % args.eval_freq == 0 or r == args.rounds:
            acc = evaluate(server.global_model, test_loader, device)
            print(f"[FedRL-AC][{name}][{r}/{args.rounds}] Test Acc={acc*100:.2f}%, Reward={reward:.4f}")
            history['round'].append(r)
            history['accuracy'].append(acc)
            history['reward'].append(reward)

    # Lưu kết quả và đồ thị tương tự
    torch.save(server.global_model.state_dict(), os.path.join(out_dir, 'model.pth'))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)

    plt.figure()
    plt.plot(history['round'], history['accuracy'])
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title(f'FedRL-AC on {name}')
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    plt.close()
    plt.figure()
    plt.plot(history['round'], history['reward'])
    plt.xlabel('Round')
    plt.ylabel('Reward')
    plt.title(f'FedRL-AC Reward on {name}')
    plt.savefig(os.path.join(out_dir, 'reward.png'))
    plt.close()

class PPOMemory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.next_states = []

    def add(self, state, action, logprob, reward, next_state=None):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        if next_state is not None:
            self.next_states.append(next_state)

class PPOServer:
    def __init__(self, model_fn, args, sizes):
        self.device = args.device
        self.global_model = model_fn().to(self.device)
        self.data_fracs = torch.tensor(sizes, dtype=torch.float, device=self.device)
        self.num_clients = len(sizes)
        self.num_opt_types = args.num_opt_types

        # Actor network (chọn optimizer cho mỗi client)
        input_dim = len(sizes) + 1  # data_fracs + prev_acc
        output_dim = len(sizes) * args.num_opt_types
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim)
        ).to(self.device)

        # Critic network (ước tính value)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        ).to(self.device)

        # Optimizer cho cả actor và critic
        self.optim = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=args.policy_lr
        )

        # Siêu tham số PPO
        self.eps_clip = getattr(args, 'ppo_clip', 0.2)
        self.K_epochs = getattr(args, 'ppo_epochs', 4)
        self.gamma = getattr(args, 'ppo_gamma', 0.99)

        self.memory = PPOMemory()

    def select(self, state):
        """Chọn action (optimizer) cho mỗi client"""
        logits = self.actor(state)
        # Reshape để có shape (num_clients, num_opt_types)
        logits = logits.view(self.num_clients, self.num_opt_types)
        
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        # Sample action cho mỗi client
        acts = dist.sample()
        logp = dist.log_prob(acts).sum()
        
        return acts.cpu().tolist(), logp

    def aggregate(self, client_models):
        """Aggregate models từ clients"""
        gdict = self.global_model.state_dict()
        for k in gdict: 
            gdict[k] = torch.zeros_like(gdict[k])
        
        total = self.data_fracs.sum().item()
        for cm, w in zip(client_models, self.data_fracs.tolist()):
            for k, v in cm.state_dict().items():
                gdict[k] += v.to(self.device) * (w / total)
        
        self.global_model.load_state_dict(gdict)

    def _compute_returns_and_advantages(self):
        """Tính returns và advantages cho PPO"""
        returns = []
        G = 0
        for r in reversed(self.memory.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Tính values và advantages
        states = torch.cat(self.memory.states, dim=0)
        values = self.critic(states).squeeze()
        advantages = returns - values.detach()
        
        return returns, advantages

    def update(self):
        """Cập nhật actor và critic networks"""
        if len(self.memory.states) == 0:
            return
            
        returns, advantages = self._compute_returns_and_advantages()
        old_logprobs = torch.stack(self.memory.logprobs).detach()
        states = torch.cat(self.memory.states, dim=0)
        actions = torch.tensor(self.memory.actions, dtype=torch.long, device=self.device)

        for _ in range(self.K_epochs):
            # Forward pass
            logits = self.actor(states)
            logits = logits.view(-1, self.num_clients, self.num_opt_types)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            logprobs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1).mean()
            values = self.critic(states).squeeze()

            # PPO loss
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values, returns)
            
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            # Backward pass
            self.optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 
                1.0
            )
            self.optim.step()

        self.memory.clear()

def train_fedrl_ppo(name, dataset, args, split_idxs, out_dir):
    """Train FedRL với PPO - PHIÊN BẢN SỬA LỖI"""
    os.makedirs(out_dir, exist_ok=True)
    device = args.device

    # Chuẩn bị DataLoader và sizes
    loaders, sizes = [], []
    for idxs in split_idxs:
        ds = ClientDataset(dataset['train'], idxs)
        sizes.append(len(ds))
        loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=True))
    
    test_loader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False)

    # Khởi tạo server
    server = PPOServer(lambda: SimpleCNN(), args, sizes)
    prev_acc = evaluate(server.global_model, test_loader, device)
    
    # History để lưu metrics
    history = {'round': [], 'accuracy': [], 'reward': []}

    print(f"[FedRL-PPO][{name}] Bắt đầu training với {len(loaders)} clients...")
    
    for r in range(1, args.rounds + 1):
        # Tạo state: data_fracs chuẩn hóa + prev_acc
        state = torch.cat([
            server.data_fracs / server.data_fracs.sum(),
            torch.tensor([prev_acc], device=device)
        ]).unsqueeze(0)

        # Chọn actions (optimizers) cho các clients
        acts, logp = server.select(state)

        # Local training với optimizer được chọn
        client_models = []
        for i, loader in enumerate(loaders):
            lm = deepcopy(server.global_model).to(device)
            opt_fn = args.optimizer_fns[acts[i]]
            opt = opt_fn(lm.parameters())
            
            lm.train()
            for _ in range(args.local_epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = F.cross_entropy(lm(x), y)
                    loss.backward()
                    opt.step()
            
            client_models.append(lm)

        # Aggregate models
        server.aggregate(client_models)
        
        # Tính accuracy sau aggregation
        curr_acc = evaluate(server.global_model, test_loader, device)
        
        # Định nghĩa reward
        reward = (curr_acc - prev_acc) + 0.1 * curr_acc
        
        # Lưu experience vào memory
        server.memory.add(state, acts, logp, reward)

        # Cập nhật PPO nếu đủ experience hoặc cuối epoch
        if len(server.memory.states) >= args.eval_freq or r == args.rounds:
            server.update()

        # Logging và evaluation
        if r % args.eval_freq == 0 or r == args.rounds:
            print(f"[FedRL-PPO][{name}][{r}/{args.rounds}] "
                  f"Acc={curr_acc*100:.2f}% (Δ={curr_acc-prev_acc:.4f}) "
                  f"Reward={reward:.4f}")
            
            history['round'].append(r)
            history['accuracy'].append(curr_acc)
            history['reward'].append(reward)
        
        # Cập nhật prev_acc cho round tiếp theo
        prev_acc = curr_acc

    # Lưu kết quả SAU KHI HOÀN THÀNH training
    print(f"[FedRL-PPO][{name}] Lưu kết quả vào {out_dir}")
    
    # Lưu model
    torch.save(server.global_model.state_dict(), 
               os.path.join(out_dir, 'model.pth'))
    
    # Lưu metrics
    pd.DataFrame(history).to_csv(
        os.path.join(out_dir, 'metrics.csv'), 
        index=False
    )
    
    # Vẽ accuracy chart
    plt.figure(figsize=(10, 6))
    plt.plot(history['round'], history['accuracy'], marker='o')
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy')
    plt.title(f'FedRL-PPO Accuracy on {name}')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'accuracy.png'), dpi=150)
    plt.close()
    
    # Vẽ reward chart
    plt.figure(figsize=(10, 6))
    plt.plot(history['round'], history['reward'], marker='s', color='red')
    plt.xlabel('Round')
    plt.ylabel('Reward')
    plt.title(f'FedRL-PPO Reward on {name}')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'reward.png'), dpi=150)
    plt.close()
    
    print(f"[FedRL-PPO][{name}] Hoàn thành! Final accuracy: {curr_acc*100:.2f}%")

# -------------------- Main --------------------
if __name__ == "__main__":
    # 1) Đặt seed để kết quả có thể tái lập qua các lần chạy
    set_seed(42)
    # 2) Chọn thiết bị tính toán: ưu tiên GPU nếu có, ngược lại dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) Định nghĩa class Args chứa siêu tham số cho toàn bộ experiment
    class Args:
        num_clients = 100                  # số lượng client (máy con) tham gia FL
        local_epochs = 3                   # số epoch huấn luyện local mỗi vòng
        rounds = 1000                       # số vòng giao tiếp giữa server và client
        batch_size = 32                    # kích thước batch cho local training
        lr = 0.01                          # learning rate cho optimizer SGD ở FedAvg/FedRL-AC
        policy_lr = 1e-3                   # learning rate cho policy network (FedRL-DRL/AC)
        value_lr = 1e-3                    # learning rate cho value network (FedRL-AC)
        num_opt_types = 3                  # số loại optimizer (SGD, Adam, RMSprop)
        optimizer_fns = [                  # danh sách hàm khởi tạo optimizer
            lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
            lambda p: optim.Adam(p, lr=0.001),
            lambda p: optim.RMSprop(p, lr=0.005, alpha=0.9)
        ]
        eval_freq = 10                     # tần suất (vòng) đánh giá global model trên test set
        device = device                    # lưu lại device dùng chung
        alpha = 0.2                        # tham số alpha cho Dirichlet partition
        main_ratio = 0.6                   # tỉ lệ client chính cho Cluster partitions
        labels_per_client = 2              # số nhãn gán cho mỗi client chính
        save_root = "./results_100clients_1000round"           # thư mục gốc lưu kết quả

        # Thêm hyperparameters cho PPO
        ppo_clip = 0.2      # PPO clipping parameter
        ppo_epochs = 4      # số epochs update PPO mỗi lần
        ppo_gamma = 0.99    # discount factor
        
        # Có thể điều chỉnh eval_freq để PPO update thường xuyên hơn
        eval_freq = 5  # thay vì 10

    args = Args()

    # 4) Thiết lập transform: chuyển ảnh về tensor và chuẩn hoá về [-1,1]
    transforms_cfg = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 5) Chọn dataset cần chạy (ở đây chỉ dùng FashionMNIST) 
    # datasets_cfg = [("FashionMNIST", datasets.FashionMNIST)]
    datasets_cfg = [("MNIST", datasets.MNIST)] 

    # 6) Định nghĩa các hàm phân chia dữ liệu Non-IID
    partitions = {
        # IID: trả về list các list chỉ số
        "IID": lambda train_ds: [
            list(idxs) 
            for idxs in mnist_iid(train_ds, args.num_clients).values()
        ],

        # non‑IID: mỗi client 2 shard (300 shards × 200 ảnh)
        "NonIID": lambda train_ds: [
            idxs.astype(int).tolist() 
            for idxs in mnist_noniid(train_ds, args.num_clients).values()
        ],

        # non‑IID‑unequal: số lượng shard không đều (1200 shards × 50 ảnh)
        "NonIID-Unequal": lambda train_ds: [
            idxs.astype(int).tolist() 
            for idxs in mnist_noniid_unequal(train_ds, args.num_clients).values()
        ]
    }

    # 7) Vòng lặp chính: với mỗi dataset và mỗi kiểu partition
    for name, cls in datasets_cfg:
        print(f"\n=== Dataset: {name} ===")
        # 7.1) Load train/test Dataset
        train_ds = cls(root="./data", train=True, download=True, transform=transforms_cfg)
        test_ds = cls(root="./data", train=False, download=True, transform=transforms_cfg)
        dataset = {"train": train_ds, "test": test_ds}

        for part_name, fn in partitions.items():
            print(f"\n--- Partition: {part_name} ---")

            # Tạo thư mục lưu splits
            base = os.path.join(args.save_root, name, part_name)
            os.makedirs(base, exist_ok=True)
            splits_path = os.path.join(base, "client_splits.pkl")

            # Nếu đã có file splits thì load, không thì compute + save
            if os.path.exists(splits_path):
                with open(splits_path, "rb") as f:
                    idxs = pickle.load(f)
                print(f"Loaded client splits từ {splits_path}")
            else:
                idxs = fn(train_ds)
                # đảm bảo mỗi client có ít nhất 1 mẫu
                for i in range(len(idxs)):
                    if not idxs[i]:
                        idxs[i] = [random.randrange(len(train_ds))]
                with open(splits_path, "wb") as f:
                    pickle.dump(idxs, f)
                print(f"Saved client splits vào {splits_path}")

            
            # vẽ và lưu phân bố data per client × label ---
            # Tính ma trận phân bố: hàng = client, cột = label
            num_clients = len(idxs)
            # với FashionMNIST có 10 label, nếu bạn dùng dataset khác, thay 10 bằng số label tương ứng
            num_labels = 10
            dist = np.zeros((num_clients, num_labels), dtype=int)
            for c, id_list in enumerate(idxs):
                for idx in id_list:
                    _, y = train_ds[idx]
                    dist[c, y] += 1

            # Vẽ heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(dist, aspect='auto')
            plt.colorbar(label='Số mẫu')
            plt.xlabel('Label')
            plt.ylabel('Client')
            plt.title(f'Data distribution for {part_name}')
            # Tạo thư mục nếu chưa có
            base = os.path.join(args.save_root, name, part_name)
            os.makedirs(base, exist_ok=True)
            # Lưu ảnh
            plt.savefig(os.path.join(base, 'data_distribution.png'))
            plt.close()

            # 7.4) Chạy FedAvg, FedRL-DRL và FedRL-AC theo thứ tự
            train_fedavg(name, dataset, args, idxs, os.path.join(base, "FedAvg"))
            # train_fedrl_drl(name, dataset, args, idxs, os.path.join(base, "FedRL-DRL"))
            train_fedrl_ac(name, dataset, args, idxs, os.path.join(base, "FedRL-AC"))
            train_fedrl_ppo(name, dataset, args, idxs, os.path.join(base, "FedRL-PPO"))
