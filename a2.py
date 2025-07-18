import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict, deque, namedtuple, defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
import gc

# ---------- Utils ----------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# IID partition
def iid_partition(dataset, num_clients):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return list(np.array_split(indices, num_clients))

# Non-IID partitions
def pareto_partition(dataset, num_clients, alpha=0.5):
    label2idx = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label2idx[label].append(idx)
    clients_idx = [[] for _ in range(num_clients)]
    for idxs in label2idx.values():
        np.random.shuffle(idxs)
        props = np.random.dirichlet([alpha] * num_clients)
        raw = props * len(idxs)
        splits = np.floor(raw).astype(int)
        splits[-1] += len(idxs) - splits.sum()
        start = 0
        for i, cnt in enumerate(splits):
            end = start + cnt
            clients_idx[i].extend(idxs[start:end])
            start = end
    return clients_idx

def cluster_equal_partition(dataset, num_clients, main_ratio=0.6, labels_per_client=2):
    label2idx = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label2idx[label].append(idx)
    main_clients = int(main_ratio * num_clients)
    labels = list(label2idx.keys())
    client_labels = [list(np.random.choice(labels, labels_per_client, replace=False)) for _ in range(main_clients)]
    client_labels += [[] for _ in range(num_clients - main_clients)]
    clients_idx = [[] for _ in range(num_clients)]
    for label, idxs in label2idx.items():
        np.random.shuffle(idxs)
        clients_with = [i for i, labs in enumerate(client_labels) if label in labs]
        if not clients_with:
            continue
        cnt = len(idxs) // len(clients_with)
        for j, cli in enumerate(clients_with):
            start = j * cnt
            end = start + cnt if j < len(clients_with) - 1 else len(idxs)
            clients_idx[cli].extend(idxs[start:end])
    return clients_idx

def cluster_non_equal_partition(dataset, num_clients, main_ratio=0.6, labels_per_client=2, alpha=0.5):
    label2idx = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label2idx[label].append(idx)
    main_clients = int(main_ratio * num_clients)
    labels = list(label2idx.keys())
    client_labels = [list(np.random.choice(labels, labels_per_client, replace=False)) for _ in range(main_clients)]
    client_labels += [[] for _ in range(num_clients - main_clients)]
    clients_idx = [[] for _ in range(num_clients)]
    for label, idxs in label2idx.items():
        np.random.shuffle(idxs)
        clients_with = [i for i, labs in enumerate(client_labels) if label in labs]
        if not clients_with:
            continue
        props = np.random.dirichlet([alpha] * len(clients_with))
        raw = props * len(idxs)
        splits = np.floor(raw).astype(int)
        splits[-1] += len(idxs) - splits.sum()
        start = 0
        for cnt, cli in zip(splits, clients_with):
            end = start + cnt
            clients_idx[cli].extend(idxs[start:end])
            start = end
    return clients_idx

# ---------- DRL Policy ----------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_clients, num_opt_types, hidden_dim=128):
        super().__init__()
        self.num_clients = num_clients
        self.num_opt_types = num_opt_types
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.opt_head = nn.Linear(hidden_dim, num_clients * num_opt_types)

    def forward(self, state):
        h = self.shared(state)
        logits = self.opt_head(h)
        logits = logits.view(self.num_clients, self.num_opt_types)
        return F.softmax(logits, dim=-1)

# ---------- Federated DRL Server ----------
Transition = namedtuple('Transition', ('state', 'actions', 'reward', 'next_state'))

class FedRLServer:
    def __init__(self, model_fn, num_clients, num_opt_types, data_fracs, args):
        self.device = args.device
        self.K = num_clients
        self.M = num_opt_types
        self.data_fracs = torch.tensor(data_fracs, dtype=torch.float32, device=self.device)
        self.global_model = model_fn().to(self.device)
        self.policy = PolicyNetwork(args.state_dim, num_clients, num_opt_types, args.policy_hidden).to(self.device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=args.policy_lr)
        self.buffer = deque(maxlen=args.buffer_size)
        self.var_lambda = args.var_lambda
        self.server_lambda = args.server_lambda

    def compute_state(self, losses, global_acc):
        ls = torch.tensor(losses, dtype=torch.float32, device=self.device)
        norm_ls = ls / (ls.sum() + 1e-8)
        acc_tensor = torch.tensor([global_acc], dtype=torch.float32, device=self.device)
        return torch.cat([norm_ls, self.data_fracs, acc_tensor], dim=0)

    def select_action(self, state, explore=True):
        self.policy.eval()
        with torch.no_grad():
            probs = self.policy(state)
            if explore:
                dist = Categorical(probs)
                actions = dist.sample()
            else:
                actions = torch.argmax(probs, dim=-1)
        return actions.cpu().tolist()

    def push_transition(self, state, actions, reward, next_state):
        self.buffer.append(Transition(state.cpu(), actions, reward, next_state.cpu()))

    def train_offline(self, batch_size):
        if len(self.buffer) < batch_size:
            return
        
        self.policy.train()
        batch = random.sample(self.buffer, batch_size)
        total_loss = 0
        
        for state, actions, reward, next_state in batch:
            state = state.to(self.device)
            probs = self.policy(state)
            dist = Categorical(probs)
            act_tensor = torch.tensor(actions, device=self.device)
            logp = dist.log_prob(act_tensor).sum()
            total_loss += -logp * reward
            
        loss = total_loss / batch_size
        self.policy_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_opt.step()

    def aggregate(self, client_models):
        """FedAvg aggregation weighted by data fractions"""
        global_dict = self.global_model.state_dict()
        
        # Initialize aggregated parameters
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
        
        # Weighted average
        for i, client_model in enumerate(client_models):
            client_dict = client_model.state_dict()
            for key in global_dict.keys():
                global_dict[key] += client_dict[key].to(self.device) * self.data_fracs[i]
        
        self.global_model.load_state_dict(global_dict)

# ---------- Model ----------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*5*5, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x): 
        return self.net(x)

def model_fn(): 
    return SimpleCNN()

# ---------- Federated Training ----------
def train_fedrl(args, name, cls):
    # Validate arguments
    assert args.var_lambda >= 0, "var_lambda must be non-negative"
    assert args.server_lambda >= 0, "server_lambda must be non-negative"
    assert args.num_clients > 0, "num_clients must be positive"
    assert args.local_epochs > 0, "local_epochs must be positive"
    
    transform = transforms.ToTensor()
    train_set = cls(args.data_dir, train=True, download=True, transform=transform)
    test_set = cls(args.data_dir, train=False, download=True, transform=transform)

    # Data partitioning
    if args.partition == 'iid':
        client_idxs = iid_partition(train_set, args.num_clients)
    elif args.partition == 'dirichlet':
        client_idxs = pareto_partition(train_set, args.num_clients, args.alpha)
    elif args.partition == 'CE':
        client_idxs = cluster_equal_partition(train_set, args.num_clients, args.main_ratio, args.labels_per_client)
    elif args.partition == 'CN':
        client_idxs = cluster_non_equal_partition(train_set, args.num_clients, args.main_ratio, args.labels_per_client, args.alpha)
    else:
        raise ValueError(f'Unknown partition: {args.partition}')

    # Ensure no empty client datasets
    for i in range(len(client_idxs)):
        if len(client_idxs[i]) == 0:
            client_idxs[i] = [random.randrange(len(train_set))]

    data_sizes = [len(idxs) for idxs in client_idxs]
    total = sum(data_sizes)
    data_fracs = [s/total for s in data_sizes]

    # Create save directory
    save_dir = os.path.join(args.save_dir, f"{name}_{args.partition}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot data distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(args.num_clients), data_sizes)
    plt.title(f'{name} ({args.partition}) Data Distribution')
    plt.xlabel('Client ID')
    plt.ylabel('Number of Samples')
    plt.savefig(os.path.join(save_dir, 'data_dist.png'))
    plt.close()

    # Create data loaders
    loaders = [DataLoader(Subset(train_set, idxs), batch_size=args.batch_size, shuffle=True) 
               for idxs in client_idxs]
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Initialize server
    server = FedRLServer(model_fn, args.num_clients, args.num_opt_types, data_fracs, args)
    prev_acc = evaluate(server.global_model, test_loader, args)
    history = {'pre': [], 'post': [], 'acc': [], 'cr': [], 'sr': []}

    print(f"Initial accuracy: {prev_acc:.4f}")

    # Training loop
    for r in range(args.rounds):
        try:
            print(f"[{name}_{args.partition}] Round {r+1}/{args.rounds}")
            
            # Compute pre-update losses
            pre_losses = []
            server.global_model.eval()
            with torch.no_grad():
                for loader in loaders:
                    loss_sum = 0
                    sample_count = 0
                    for x, y in loader:
                        x, y = x.to(args.device), y.to(args.device)
                        logits = server.global_model(x)
                        loss = F.cross_entropy(logits, y, reduction='sum')
                        loss_sum += loss.item()
                        sample_count += x.size(0)
                    pre_losses.append(loss_sum / sample_count)

            # Compute state and select actions
            state = server.compute_state(pre_losses, prev_acc)
            actions = server.select_action(state, explore=True)

            # Local training
            post_losses = []
            client_models = []
            
            for i, loader in enumerate(loaders):
                # Create local model copy
                local_model = deepcopy(server.global_model)
                local_model.train()
                
                # Initialize optimizer
                optimizer = args.optimizer_fns[actions[i]](local_model.parameters())
                
                # Local training
                for epoch in range(args.local_epochs):
                    for x, y in loader:
                        x, y = x.to(args.device), y.to(args.device)
                        optimizer.zero_grad()
                        logits = local_model(x)
                        loss = F.cross_entropy(logits, y)
                        loss.backward()
                        optimizer.step()
                
                # Compute post-update loss
                local_model.eval()
                with torch.no_grad():
                    loss_sum = 0
                    sample_count = 0
                    for x, y in loader:
                        x, y = x.to(args.device), y.to(args.device)
                        logits = local_model(x)
                        loss = F.cross_entropy(logits, y, reduction='sum')
                        loss_sum += loss.item()
                        sample_count += x.size(0)
                    post_losses.append(loss_sum / sample_count)
                
                # Move to CPU to save GPU memory
                local_model.cpu()
                client_models.append(local_model)

            # Server aggregation
            server.aggregate(client_models)
            
            # Evaluate global model
            curr_acc = evaluate(server.global_model, test_loader, args)

            # Compute rewards
            pre_tensor = torch.tensor(pre_losses, device=args.device)
            post_tensor = torch.tensor(post_losses, device=args.device)
            improvement = pre_tensor - post_tensor
            
            # Client reward: mean improvement - variance penalty
            client_reward = improvement.mean().item() - args.var_lambda * improvement.var(unbiased=False).item()
            
            # Server reward: accuracy improvement
            server_reward = curr_acc - prev_acc
            
            # Combined reward
            total_reward = client_reward + args.server_lambda * server_reward

            # Store transition
            next_state = server.compute_state(post_losses, curr_acc)
            server.push_transition(state, actions, total_reward, next_state)
            
            # Train policy
            server.train_offline(args.batch_size)

            # Update history
            prev_acc = curr_acc
            history['pre'].append(pre_losses)
            history['post'].append(post_losses)
            history['acc'].append(curr_acc)
            history['cr'].append(client_reward)
            history['sr'].append(server_reward)

            # Print progress
            print(f"  Accuracy: {curr_acc:.4f} | Client Reward: {client_reward:.4f} | "
                  f"Server Reward: {server_reward:.4f} | Total Reward: {total_reward:.4f}")

            # Clean up memory
            del client_models
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in round {r+1}: {e}")
            continue

    # Save results
    try:
        # Save history to CSV
        history_file = os.path.join(save_dir, 'history.csv')
        with open(history_file, 'w') as f:
            f.write('round,accuracy,client_reward,server_reward,pre_losses,post_losses\n')
            for i in range(len(history['acc'])):
                pre_str = ','.join([f'{x:.6f}' for x in history['pre'][i]])
                post_str = ','.join([f'{x:.6f}' for x in history['post'][i]])
                f.write(f"{i+1},{history['acc'][i]:.6f},{history['cr'][i]:.6f},"
                       f"{history['sr'][i]:.6f},\"{pre_str}\",\"{post_str}\"\n")
        print(f"Saved history to {history_file}")

        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0, 0].plot(range(1, args.rounds+1), history['acc'], 'b-', linewidth=2)
        axes[0, 0].set_title('Test Accuracy')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)
        
        # Loss plot
        mean_losses = [np.mean(losses) for losses in history['post']]
        axes[0, 1].plot(range(1, args.rounds+1), mean_losses, 'r-', linewidth=2)
        axes[0, 1].set_title('Average Training Loss')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Rewards plot
        axes[1, 0].plot(range(1, args.rounds+1), history['cr'], 'g-', label='Client Reward', linewidth=2)
        axes[1, 0].plot(range(1, args.rounds+1), history['sr'], 'm-', label='Server Reward', linewidth=2)
        axes[1, 0].set_title('Rewards')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Loss variance plot
        loss_vars = [np.var(losses) for losses in history['post']]
        axes[1, 1].plot(range(1, args.rounds+1), loss_vars, 'orange', linewidth=2)
        axes[1, 1].set_title('Loss Variance Across Clients')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plots to {save_dir}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

# Evaluation function
def evaluate(model, loader, args):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    return correct / total

# Main execution
if __name__ == '__main__':
    class Args:
        pass
    
    args = Args()
    
    # Basic settings
    args.seed = 123
    args.data_dir = './data'
    args.save_dir = './results'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Federated learning settings
    args.num_clients = 10
    args.num_opt_types = 3
    args.optimizer_fns = [
        lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
        lambda p: optim.Adam(p, lr=0.001),
        lambda p: optim.RMSprop(p, lr=0.005, alpha=0.9),
    ]
    args.batch_size = 64
    args.local_epochs = 5
    args.rounds = 100
    
    # DRL settings
    args.policy_hidden = 128
    args.policy_lr = 1e-3
    args.buffer_size = 100
    args.var_lambda = 0.1
    args.server_lambda = 1.0
    args.state_dim = args.num_clients * 2 + 1
    
    # Data partition settings
    args.partition = 'CE'
    args.alpha = 0.5
    args.main_ratio = 0.6
    args.labels_per_client = 2
    
    print(f"Using device: {args.device}")
    print(f"Training settings: {args.num_clients} clients, {args.rounds} rounds")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Train on datasets
    datasets_to_train = [('MNIST', datasets.MNIST), ('FashionMNIST', datasets.FashionMNIST)]
    
    for name, dataset_class in datasets_to_train:
        print(f"\n{'='*50}")
        print(f"Training on {name}")
        print(f"{'='*50}")
        try:
            train_fedrl(args, name, dataset_class)
            print(f"✓ Successfully completed training on {name}")
        except Exception as e:
            print(f"✗ Failed to train on {name}: {e}")
        
        # Clean up between datasets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)