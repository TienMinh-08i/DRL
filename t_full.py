import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms, models
from collections import deque, namedtuple, defaultdict
from tqdm.auto import trange
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, classification_report

# --- 1) Seed reproducibility ---
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# --- 2) Non-IID partition (Dirichlet) ---
def pareto_partition(dataset: Dataset, num_clients: int, alpha: float = 0.5):
    label2idx = {}
    for idx, (_, label) in enumerate(dataset):
        label2idx.setdefault(label, []).append(idx)
    clients_idx = [[] for _ in range(num_clients)]
    for idxs in label2idx.values():
        np.random.shuffle(idxs)
        props = np.random.dirichlet([alpha]*num_clients)
        raw = props * len(idxs)
        splits = np.floor(raw).astype(int)
        splits[-1] += len(idxs) - splits.sum()
        start = 0
        for i, cnt in enumerate(splits):
            end = start + cnt
            clients_idx[i].extend(idxs[start:end])
            start = end
    return clients_idx

def cluster_equal_partition(dataset: Dataset,
                            num_clients: int,
                            main_ratio: float = 0.6,
                            labels_per_client: int = 2):
    """
    CE (Clustered-Equal): 
    - Chọn main_clients = int(main_ratio * num_clients) client đầu sẽ mỗi client có đúng `labels_per_client` nhãn.
    - Phân phối đều các sample của mỗi nhãn cho các client chọn nhãn đó.
    - Phần còn lại của client (num_clients - main_clients) có thể để rỗng hoặc random labels tùy bạn triển khai.
    """
    # 1) Tạo map label -> list index
    label2idx = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label2idx[label].append(idx)

    # 2) Xác định nhãn cho mỗi client
    main_clients = int(main_ratio * num_clients)
    all_labels = list(label2idx.keys())
    client_labels = []
    for i in range(main_clients):
        # random chọn labels_per_client nhãn không lặp
        client_labels.append(list(np.random.choice(all_labels, labels_per_client, replace=False)))
    # nếu muốn, có thể gán nhãn rỗng cho các client còn lại
    for i in range(main_clients, num_clients):
        client_labels.append([])

    # 3) Phân chia chỉ số
    clients_idx = [[] for _ in range(num_clients)]
    for label, idxs in label2idx.items():
        np.random.shuffle(idxs)
        # tìm các client chứa label này
        clients_with = [i for i, labs in enumerate(client_labels) if label in labs]
        if not clients_with:
            continue
        cnt_per = len(idxs) // len(clients_with)
        for j, cli in enumerate(clients_with):
            start = j * cnt_per
            end = start + cnt_per if j < len(clients_with) - 1 else len(idxs)
            clients_idx[cli].extend(idxs[start:end])
    return clients_idx


def cluster_non_equal_partition(dataset: Dataset,
                                num_clients: int,
                                main_ratio: float = 0.6,
                                labels_per_client: int = 2,
                                alpha: float = 0.5):
    """
    CN (Clustered-NonEqual):
    - Chọn main_clients = int(main_ratio * num_clients) client đầu, mỗi client có `labels_per_client` nhãn.
    - Với mỗi nhãn, phân phối số mẫu đến các client có nhãn đó theo tỉ lệ Dirichlet(alpha).
    - Client còn lại có thể để rỗng.
    """
    # 1) Tạo map label -> list index
    label2idx = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label2idx[label].append(idx)

    # 2) Xác định nhãn cho mỗi client (giống CE)
    main_clients = int(main_ratio * num_clients)
    all_labels = list(label2idx.keys())
    client_labels = []
    for i in range(main_clients):
        client_labels.append(list(np.random.choice(all_labels, labels_per_client, replace=False)))
    for i in range(main_clients, num_clients):
        client_labels.append([])

    # 3) Phân chia chỉ số với Dirichlet
    clients_idx = [[] for _ in range(num_clients)]
    for label, idxs in label2idx.items():
        np.random.shuffle(idxs)
        # clients có label này
        clients_with = [i for i, labs in enumerate(client_labels) if label in labs]
        if not clients_with:
            continue
        # lấy tỉ lệ Dirichlet
        props = np.random.dirichlet([alpha] * len(clients_with))
        raw = props * len(idxs)
        splits = np.floor(raw).astype(int)
        # điều chỉnh cho đủ tổng
        splits[-1] += len(idxs) - splits.sum()
        start = 0
        for cnt, cli in zip(splits, clients_with):
            end = start + cnt
            clients_idx[cli].extend(idxs[start:end])
            start = end
    return clients_idx

# --- 3) Model & ClientDataset ---
class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,1), nn.ReLU(),
            nn.Conv2d(32,64,3,1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(9216,128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128,num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))
    def get_embedding(self, x): return torch.flatten(self.features(x),1)

class VGG19Model(nn.Module):
    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()
        # Lấy toàn bộ phần features của VGG19
        vgg = models.vgg19(pretrained=pretrained)
        self.features = vgg.features
        # Classifier gốc: [Flatten, 4096→4096→4096→num_classes]
        # Ta giữ nguyên, chỉ thay num_classes cuối
        clf = list(vgg.classifier.children())
        clf[-1] = nn.Linear(in_features=4096, out_features=num_classes)
        self.classifier = nn.Sequential(*clf)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_embedding(self, x):
        # embedding trước lớp cuối cùng
        x = self.features(x)
        x = torch.flatten(x, 1)
        # trả về embedding 4096-dim
        return x


class ClientDataset(Dataset):
    def __init__(self, ds, idxs): self.ds, self.idxs = ds, list(idxs)
    def __len__(self): return len(self.idxs)
    def __getitem__(self,i): return self.ds[self.idxs[i]]

# --- 4) Prioritized Replay Buffer ---
Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
    def push(self, *args):
        max_p = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
            self.priorities.append(max_p)
        else:
            # overwrite oldest
            idx = len(self.buffer) % self.capacity
            self.buffer[idx] = Transition(*args)
            self.priorities[idx] = max_p
    def sample(self, batch_size, beta=0.4):
        pri = np.array(self.priorities)**self.alpha
        probs = pri / pri.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        # importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices])**(-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float)
    def update_priorities(self, indices, losses):
        for idx, loss in zip(indices, losses):
            self.priorities[idx] = abs(loss.item()) + 1e-6
    def __len__(self): return len(self.buffer)

# --- 5) Policy & Value Networks ---
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_clients):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU()
        )
        self.mu = nn.Linear(256, num_clients)
        self.log_sigma = nn.Linear(256, num_clients)
    def forward(self, x):
        h = self.fc(x)
        mu = self.mu(h)
        sigma = torch.exp(self.log_sigma(h)).clamp(0.01,1.0)
        return mu, sigma

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, num_clients):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim+num_clients,256), nn.LeakyReLU(),
            nn.Linear(256,256), nn.LeakyReLU(),
            nn.Linear(256,1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state,action],1))

# --- 6) FedRLServer with two-stage training ---
class FedRLServer:
    def __init__(self, model_fn, num_clients, args):
        self.global_model = model_fn().to(args.device)
        self.num_clients = num_clients
        self.args = args

        # state dim = num_clients * 3 features
        self.state_dim = num_clients * 3

        # actor-critic + target networks
        self.policy = PolicyNetwork(self.state_dim, num_clients).to(args.device)
        self.value  = ValueNetwork(self.state_dim, num_clients).to(args.device)
        self.target_policy = PolicyNetwork(self.state_dim, num_clients).to(args.device)
        self.target_value  = ValueNetwork(self.state_dim, num_clients).to(args.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_value.load_state_dict(self.value.state_dict())

        # optimizers
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=args.policy_lr)
        self.value_opt  = optim.Adam(self.value.parameters(),  lr=args.value_lr)

        # prioritized buffer
        self.buffer = PrioritizedReplayBuffer(args.buffer_size, alpha=args.prio_alpha)

        # RL hyperparams
        self.gamma = args.gamma
        self.tau   = args.tau

        # track metrics & for reward calc
        self.prev_avg_loss = None
        self.prev_gap      = None

    def compute_state(self, glob_losses, loc_losses, sizes):
        g = torch.tensor(glob_losses, device=self.args.device)
        l = torch.tensor(loc_losses, device=self.args.device)
        s = torch.tensor(sizes, dtype=torch.float, device=self.args.device)
        s = s / s.sum()
        return torch.cat([g,l,s]).unsqueeze(0)

    def select_action(self, state, explore=True):
        mu, sigma = self.policy(state)
        if explore:
            eps = torch.randn_like(mu)
            action = mu + sigma * eps
        else:
            action = mu
        weights = F.softmax(action, dim=1)
        return action, weights

    def aggregate(self, client_models, client_states):
        glob, loc, sizes = zip(*client_states)
        state = self.compute_state(glob, loc, sizes)
        action, weights = self.select_action(state)

        # weighted average parameters
        new_sd = {}
        for k in self.global_model.state_dict().keys():
            new_sd[k] = sum(weights[0,i] * m.state_dict()[k].to(self.args.device)
                            for i,m in enumerate(client_models))
        self.global_model.load_state_dict(new_sd)

        # compute reward: Δavg_loss + Δgap
        avg_loss = np.mean(glob)
        gap      = float(max(glob) - min(glob))
        if self.prev_avg_loss is None:
            reward = 0.0
        else:
            reward = (self.prev_avg_loss - avg_loss) + (self.prev_gap - gap)
        self.prev_avg_loss, self.prev_gap = avg_loss, gap

        # push to buffer
        self.buffer.push(state, action, reward, None, False)
        return weights.squeeze().detach().cpu().numpy()

    def train_offline(self):
        for _ in range(self.args.update_steps):
            if len(self.buffer) < self.args.batch_size_rl:
                return
            batch, idxs, is_weights = self.buffer.sample(self.args.batch_size_rl, beta=self.args.prio_beta)
            # unpack
            states = torch.cat([t.state for t in batch])
            actions= torch.cat([t.action for t in batch])
            rewards= torch.tensor([t.reward for t in batch], device=self.args.device).unsqueeze(1)
            # Critic update
            self.value_opt.zero_grad()
            values = self.value(states, F.softmax(actions.detach(),1))
            # compute targets
            targets = rewards  # no next_state in our design
            td_errors = (values - targets).squeeze(1)
            value_loss = (is_weights.to(self.args.device) * td_errors.pow(2)).mean()
            value_loss.backward()
            self.value_opt.step()
            # Actor update
            self.policy_opt.zero_grad()
            mu, _ = self.policy(states)
            w = F.softmax(mu,1)
            policy_loss = -self.value(states, w).mean()
            policy_loss.backward()
            self.policy_opt.step()
            # soft update targets
            for p, tp in zip(self.policy.parameters(), self.target_policy.parameters()):
                tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
            for p, tv in zip(self.value.parameters(), self.target_value.parameters()):
                tv.data.copy_(self.tau*p.data + (1-self.tau)*tv.data)
            # update priorities
            self.buffer.update_priorities(idxs, td_errors.detach().abs())

# --- 7) Training Loop with Test Accuracy ---
def train_fedrl(train_ds, test_ds, args, model_fn, output_prefix):
    """
    train_ds, test_ds: Dataset objects
    args: arguments object
    model_fn: factory function for model instantiation
    output_prefix: prefix (e.g. dataset name) for output directory and filenames
    """


    # 0) Tạo thư mục output
    os.makedirs(output_prefix, exist_ok=True)

    # 1) Phân vùng dữ liệu non-IID và lưu phân bố (ảnh)
    # clients_idx = pareto_partition(train_ds, args.num_clients, args.alpha)
    clients_idx = cluster_equal_partition(train_ds, args.num_clients, main_ratio=0.6, labels_per_client=2)
    # clients_idx = cluster_non_equal_partition(train_ds, args.num_clients, main_ratio=0.6, labels_per_client=2, alpha=args.alpha)
    
    for i, idxs in enumerate(clients_idx):
        if len(idxs) == 0:
            # ví dụ: gán ngẫu nhiên 1 sample từ toàn bộ train_ds
            clients_idx[i] = [np.random.choice(range(len(train_ds)))]

    plt.figure()
    plt.bar(range(args.num_clients), [len(idxs) for idxs in clients_idx])
    plt.title('Data per client')
    plt.savefig(os.path.join(output_prefix, f'{output_prefix}_data_dist.png'))
    plt.close()

    # 2) Khởi tạo server
    server = FedRLServer(model_fn, args.num_clients, args)
    # Khởi tạo history cho RL metrics
    server.reward_history = []
    server.value_loss_history = []
    server.policy_loss_history = []

    # 3) Khởi tạo DataLoaders
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    train_loaders = [
        DataLoader(ClientDataset(train_ds, idxs), batch_size=args.batch_size, shuffle=True)
        for idxs in clients_idx
    ]

    # 4) Training qua các rounds, lưu accuracy
    acc_list = []
    best_acc, best_state = 0.0, None
    for rnd in range(1, args.num_rounds + 1):
        client_models, client_states = [], []
        for loader in train_loaders:
            # instantiate local model
            model_c = model_fn().to(args.device)
            model_c.load_state_dict(server.global_model.state_dict())
            opt_c = optim.SGD(model_c.parameters(), lr=args.lr, momentum=args.momentum)
            # local training
            model_c.train()
            for _ in range(args.local_epochs):
                for x, y in loader:
                    x, y = x.to(args.device), y.to(args.device)
                    opt_c.zero_grad()
                    loss = F.cross_entropy(model_c(x), y)
                    loss.backward()
                    opt_c.step()
            # đánh giá loss
            model_c.eval()
            total_loc = total_glob = cnt_loc = cnt_glob = 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(args.device), y.to(args.device)
                    preds = model_c(x)
                    total_loc += F.cross_entropy(preds, y, reduction='sum').item()
                    cnt_loc += y.size(0)
                for x, y in DataLoader(loader.dataset, batch_size=args.batch_size):
                    x, y = x.to(args.device), y.to(args.device)
                    total_glob += F.cross_entropy(model_c(x), y, reduction='sum').item()
                    cnt_glob += y.size(0)
            client_models.append(model_c)
            client_states.append((total_glob/cnt_glob, total_loc/cnt_loc, len(loader.dataset)))
        # aggregate & RL update
        weights = server.aggregate(client_models, client_states)
        # train_offline cập nhật và lưu losses
        for _ in range(args.update_steps):
            if len(server.buffer) < args.batch_size_rl:
                break
            # gọi train_offline tách để lấy losses
            batch, idxs, is_weights = server.buffer.sample(args.batch_size_rl, beta=args.prio_beta)
            states = torch.cat([t.state for t in batch])
            actions = torch.cat([t.action for t in batch])
            rewards = torch.tensor([t.reward for t in batch], device=args.device).unsqueeze(1)
            # critic
            server.value_opt.zero_grad()
            values = server.value(states, F.softmax(actions.detach(),1))
            td_errors = (values - rewards).squeeze(1)
            value_loss = (is_weights.to(args.device) * td_errors.pow(2)).mean()
            value_loss.backward()
            server.value_opt.step()
            # actor
            server.policy_opt.zero_grad()
            mu, _ = server.policy(states)
            w = F.softmax(mu,1)
            policy_loss = -server.value(states, w).mean()
            policy_loss.backward()
            server.policy_opt.step()
            # soft update targets
            for p, tp in zip(server.policy.parameters(), server.target_policy.parameters()):
                tp.data.copy_(server.tau*p.data + (1-server.tau)*tp.data)
            for p, tv in zip(server.value.parameters(), server.target_value.parameters()):
                tv.data.copy_(server.tau*p.data + (1-server.tau)*tv.data)
            server.buffer.update_priorities(idxs, td_errors.detach().abs())
            # lưu history
            server.value_loss_history.append(value_loss.item())
            server.policy_loss_history.append(policy_loss.item())
        # test
        server.global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(args.device), y.to(args.device)
                preds = server.global_model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct/total
        acc_list.append(acc)
        server.reward_history.append(server.prev_avg_loss - acc if server.prev_avg_loss is not None else 0)
        print(f"Round {rnd}/{args.num_rounds} — Test Acc: {acc*100:.2f}%")
        if acc > best_acc:
            best_acc, best_state = acc, server.global_model.state_dict().copy()
    # 5) Lưu best model
    server.global_model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(output_prefix, f'{output_prefix}_best.pth'))
    # 6) Vẽ và lưu các biểu đồ (ảnh)
    # accuracy curve
    plt.figure(); plt.plot([a*100 for a in acc_list]); plt.xlabel('Round'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy per Round')
    plt.savefig(os.path.join(output_prefix, f'{output_prefix}_accuracy_curve.png')); plt.close()
    # confusion matrix...
    y_true,y_pred = [],[]
    server.global_model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(args.device),y.to(args.device)
            p = server.global_model(x).argmax(1)
            y_true.extend(y.cpu().numpy()); y_pred.extend(p.cpu().numpy())
    cm = confusion_matrix(y_true,y_pred)
    fig,ax = plt.subplots(figsize=(10,8))
    im=ax.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
    ax.figure.colorbar(im,ax=ax)
    ax.set(title='Confusion Matrix',ylabel='True label',xlabel='Pred label')
    thresh=cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j,i,f'{cm[i,j]:d}',ha='center',va='center',color='white' if cm[i,j]>thresh else 'black')
    fig.savefig(os.path.join(output_prefix, f'{output_prefix}_confusion_matrix.png')); plt.close(fig)
    # classification report image
    report = classification_report(y_true,y_pred,digits=4)
    fig,ax=plt.subplots(figsize=(8,len(report.split('\n'))*0.3))
    ax.text(0,0,report,family='monospace');ax.axis('off')
    fig.savefig(os.path.join(output_prefix, f'{output_prefix}_classification_report.png'),bbox_inches='tight');plt.close(fig)
    # 7) Ghi giá trị RL history vào file txt
    with open(os.path.join(output_prefix, 'rl_reward_history.txt'),'w') as f:
        for r in server.reward_history: f.write(f"{r}\n")
    with open(os.path.join(output_prefix, 'value_loss_history.txt'),'w') as f:
        for vl in server.value_loss_history: f.write(f"{vl}\n")
    with open(os.path.join(output_prefix, 'policy_loss_history.txt'),'w') as f:
        for pl in server.policy_loss_history: f.write(f"{pl}\n")
    return server


# --- 1) Định nghĩa model_fn cho từng dataset ---
def get_cnn10():
    return CNNModel(num_classes=10)

def get_cnn_fashion():
    return CNNModel(num_classes=10)

def get_vgg100():
    return VGG19Model(num_classes=100, pretrained=False)

# --- 2) Khai báo cấu hình cho mỗi bộ ---
dataset_configs = [
    {
        "name": "MNIST",
        "class": datasets.MNIST,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        "num_classes": 10,
        "model_fn": get_cnn10
    },
    {
        "name": "FashionMNIST",
        "class": datasets.FashionMNIST,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ]),
        "num_classes": 10,
        "model_fn": get_cnn_fashion
    }
    # },
    # {
    #     "name": "CIFAR100",
    #     "class": datasets.CIFAR100,
    #     "transform": transforms.Compose([
    #         transforms.Resize(224),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=(0.485,0.456,0.406),
    #             std=(0.229,0.224,0.225)
    #         )
    #     ]),
    #     "num_classes": 100,
    #     "model_fn": get_vgg100
    # }
]


# --- 3) Main loop chạy lần lượt từng config ---
def main():
    # args chung cho tất cả
    class Args:
        data_path    = './data'
        alpha        = 0.5
        num_clients  = 10
        local_epochs = 3
        num_rounds   = 100
        batch_size   = 64
        lr           = 0.01
        momentum     = 0.5
        policy_lr    = 1e-4
        value_lr     = 1e-3
        gamma        = 0.99
        tau          = 0.02
        buffer_size  = 10000
        batch_size_rl= 64
        prio_alpha   = 0.6
        prio_beta    = 0.4
        update_steps = 10
        device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = Args()
    set_seed(42)

    for cfg in dataset_configs:
        print(f"\n=== Training on {cfg['name']} ===")
        # load data
        train_ds = cfg["class"](
            args.data_path, train=True,
            download=True, transform=cfg["transform"]
        )
        test_ds = cfg["class"](
            args.data_path, train=False,
            download=True, transform=cfg["transform"]
        )
        print(f"{cfg['name']}: Train={len(train_ds)}, Test={len(test_ds)}")

        # chạy FedRL với model_fn tương ứng
        server = FedRLServer(cfg["model_fn"], args.num_clients, args)
        train_fedrl(train_ds, test_ds, args, model_fn=cfg['model_fn'], output_prefix = cfg['name'] + "_1")

        # (Tuỳ chọn) lưu checkpoint
        torch.save(
            server.global_model.state_dict(),
            f"fedrl_{cfg['name']}.pth"
        )
if __name__ == '__main__':
    main()