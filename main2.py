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

def dirichlet_partition(dataset, num_clients, alpha=0.5):
    # 1) Gom chỉ số mẫu theo nhãn
    label2idx = defaultdict(list)
    for i, (_, y) in enumerate(dataset):
        label2idx[y].append(i)

    # 2) Tạo danh sách rỗng cho mỗi client
    clients = [[] for _ in range(num_clients)]

    # 3) Với mỗi nhãn, chia tỉ lệ cho mỗi client theo Dirichlet(alpha)
    for idxs in label2idx.values():
        np.random.shuffle(idxs)
        props = np.random.dirichlet([alpha] * num_clients)  # tỉ lệ Dirichlet
        counts = (props * len(idxs)).astype(int)
        # Đảm bảo tổng counts == len(idxs)
        counts[-1] += len(idxs) - counts.sum()

        ptr = 0
        for c, n in enumerate(counts):
            clients[c].extend(idxs[ptr:ptr + n])
            ptr += n

    return clients

def cluster_equal_partition(dataset, num_clients, main_ratio=0.6, labels_per_client=2):
    M = int(main_ratio * num_clients)  # Số client được gán nhãn chính thức
    labels = list({y for _, y in dataset})

    # với M client đầu, mỗi client lấy ngẫu nhiên labels_per_client nhãn chính
    client_labels = [random.sample(labels, min(labels_per_client, len(labels)))
                     for _ in range(M)]
    # phần còn lại không có nhãn chính ⇒ sẽ nhận các nhãn ngẫu nhiên
    client_labels += [[] for _ in range(num_clients - M)]

    # Gom chỉ số mẫu theo nhãn
    label2idx = defaultdict(list)
    for i, (_, y) in enumerate(dataset):
        label2idx[y].append(i)

    clients = [[] for _ in range(num_clients)]

    # Duyệt từng nhãn y, chia đều chỉ số idxs cho các client có nhãn chính
    for y, idxs in label2idx.items():
        np.random.shuffle(idxs)
        # Tìm client nào có y trong client_labels
        targets = [i for i, ls in enumerate(client_labels) if y in ls]
        if not targets:
            # Nếu không client nào có y, chọn ngẫu nhiên 1 client để gán
            targets = [random.randrange(num_clients)]
        per = len(idxs) // len(targets)
        for j, cli in enumerate(targets):
            start = j * per
            end = (j + 1) * per if j < len(targets) - 1 else len(idxs)
            clients[cli].extend(idxs[start:end])

    return clients

def cluster_non_equal_partition(dataset, num_clients, main_ratio=0.6, labels_per_client=2, alpha=0.5):
    M = int(main_ratio * num_clients)
    labels = list({y for _, y in dataset})
    client_labels = [random.sample(labels, min(labels_per_client, len(labels)))
                     for _ in range(M)]
    client_labels += [[] for _ in range(num_clients - M)]

    label2idx = defaultdict(list)
    for i, (_, y) in enumerate(dataset):
        label2idx[y].append(i)

    clients = [[] for _ in range(num_clients)]

    # Với mỗi nhãn y, phân phối chỉ số idxs theo tỉ lệ Dirichlet giữa các client có y
    for y, idxs in label2idx.items():
        np.random.shuffle(idxs)
        targets = [i for i, ls in enumerate(client_labels) if y in ls]
        if not targets:
            targets = [random.randrange(num_clients)]
        props = np.random.dirichlet([alpha] * len(targets))
        counts = (props * len(idxs)).astype(int)
        counts[-1] += len(idxs) - counts.sum()

        ptr = 0
        for j, cli in enumerate(targets):
            n = counts[j]
            clients[cli].extend(idxs[ptr:ptr + n])
            ptr += n

    return clients

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
        server.aggregate(client_models, sizes)

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
# -------------------- Main --------------------
if __name__ == "__main__":
    # 1) Đặt seed để kết quả có thể tái lập qua các lần chạy
    set_seed(42)
    # 2) Chọn thiết bị tính toán: ưu tiên GPU nếu có, ngược lại dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) Định nghĩa class Args chứa siêu tham số cho toàn bộ experiment
    class Args:
        num_clients = 10                   # số lượng client (máy con) tham gia FL
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
        save_root = "./results_v2"           # thư mục gốc lưu kết quả

    args = Args()

    # 4) Thiết lập transform: chuyển ảnh về tensor và chuẩn hoá về [-1,1]
    transforms_cfg = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 5) Chọn dataset cần chạy (ở đây chỉ dùng FashionMNIST) 
    datasets_cfg = [("FashionMNIST", datasets.FashionMNIST)]
    # datasets_cfg = [("MNIST", datasets.MNIST)] 

    # 6) Định nghĩa các hàm phân chia dữ liệu Non-IID
    partitions = {
        "Dirichlet": lambda ds: dirichlet_partition(ds, args.num_clients, args.alpha),
        "ClusterEq": lambda ds: cluster_equal_partition(
            ds, args.num_clients, args.main_ratio, args.labels_per_client
        ),
        "ClusterNonEq": lambda ds: cluster_non_equal_partition(
            ds, args.num_clients, args.main_ratio,
            args.labels_per_client, args.alpha
        )
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
            # train_fedavg(name, dataset, args, idxs, os.path.join(base, "FedAvg"))
            train_fedrl_drl(name, dataset, args, idxs, os.path.join(base, "FedRL-DRL"))
            # train_fedrl_ac(name, dataset, args, idxs, os.path.join(base, "FedRL-AC"))
