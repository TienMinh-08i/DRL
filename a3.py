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
from collections import defaultdict, namedtuple
from copy import deepcopy
import matplotlib.pyplot as plt
import gc
import json
import pandas as pd
import pickle
from datetime import datetime

# ---------- Utils ----------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Missing Classes ----------
class ClientDataset:
    """Dataset wrapper for client data"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

class SimpleCNN(nn.Module):
    """Simple CNN model for MNIST/FashionMNIST"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class FedAvgServer:
    """FedAvg Server implementation"""
    def __init__(self, global_model, data_fractions, device):
        self.global_model = global_model.to(device)
        self.data_fractions = data_fractions
        self.device = device
    
    def aggregate(self, client_models):
        """Aggregate client models using FedAvg"""
        global_dict = self.global_model.state_dict()
        
        # Initialize aggregated parameters
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
        
        # Weighted average
        total_weight = sum(self.data_fractions)
        for client_model, weight in zip(client_models, self.data_fractions):
            client_dict = client_model.state_dict()
            for key in global_dict.keys():
                global_dict[key] += client_dict[key] * (weight / total_weight)
        
        self.global_model.load_state_dict(global_dict)

def model_fn(num_classes=10):
    """Model factory function"""
    return SimpleCNN(num_classes)

def evaluate(model, data_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return correct / total

# ---------- Partition Functions ----------
def iid_partition(dataset, num_clients):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return list(np.array_split(indices, num_clients))

def dirichlet_partition(dataset, num_clients, alpha=0.5):
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
    
    # Đảm bảo mỗi label được assign cho ít nhất 1 client
    client_labels = []
    for i in range(main_clients):
        # Chọn labels sao cho balanced
        available_labels = [l for l in labels if sum(1 for cl in client_labels if l in cl) < len(client_labels)//len(labels) + 2]
        if len(available_labels) < labels_per_client:
            available_labels = labels
        selected = list(np.random.choice(available_labels, min(labels_per_client, len(available_labels)), replace=False))
        client_labels.append(selected)
    
    # Remaining clients get random distribution
    for i in range(num_clients - main_clients):
        client_labels.append(list(np.random.choice(labels, min(2, len(labels)), replace=False)))
    
    clients_idx = [[] for _ in range(num_clients)]
    
    # Phân phối dữ liệu
    for label, idxs in label2idx.items():
        np.random.shuffle(idxs)
        clients_with = [i for i, labs in enumerate(client_labels) if label in labs]
        
        if not clients_with:
            # Assign to random clients nếu không có ai muốn label này
            clients_with = list(np.random.choice(num_clients, min(2, num_clients), replace=False))
        
        cnt_per_client = len(idxs) // len(clients_with)
        remainder = len(idxs) % len(clients_with)
        
        start_idx = 0
        for j, cli in enumerate(clients_with):
            # Phân phối remainder đều
            cnt = cnt_per_client + (1 if j < remainder else 0)
            end_idx = start_idx + cnt
            clients_idx[cli].extend(idxs[start_idx:end_idx])
            start_idx = end_idx
    
    # Đảm bảo mỗi client có ít nhất min_samples_per_client
    min_samples_per_client = max(10, len(dataset) // (num_clients * 10))
    all_indices = list(range(len(dataset)))
    
    for i in range(num_clients):
        if len(clients_idx[i]) < min_samples_per_client:
            # Lấy thêm samples từ pool chung
            needed = min_samples_per_client - len(clients_idx[i])
            available = [idx for idx in all_indices if idx not in clients_idx[i]]
            if len(available) >= needed:
                additional = np.random.choice(available, needed, replace=False)
                clients_idx[i].extend(additional)
    
    return clients_idx

# ---------- Main Training Function ----------
def train_federated_improved(name, dataset_class, args):
    set_seed(args.seed)
    
    # Prepare data với validation split
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_ds = dataset_class(args.data_path, train=True, download=True, transform=transform)
    test_ds = dataset_class(args.data_path, train=False, download=True, transform=transform)
    
    # Tạo validation set từ training set
    val_size = int(0.1 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_ds, [train_size, val_size])
    
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    partitions = {
        'IID': iid_partition,
        'Dirichlet': lambda ds, k: dirichlet_partition(ds, k, alpha=args.alpha),
        'ClusterEq': lambda ds, k: cluster_equal_partition(ds, k, main_ratio=args.main_ratio, labels_per_client=args.labels_per_client)
    }

    results = {}
    
    for p_name, partition_fn in partitions.items():
        print(f"\n=== Training {name} with {p_name} partition ===")
        
        # Partition chỉ training subset
        clients_idx = partition_fn(train_subset, args.num_clients)
        
        # Kiểm tra và báo cáo data distribution
        data_sizes = [len(idx) for idx in clients_idx]
        print(f"Data sizes per client: min={min(data_sizes)}, max={max(data_sizes)}, avg={np.mean(data_sizes):.1f}")
        
        if min(data_sizes) == 0:
            print("Warning: Some clients have no data!")
            continue
            
        data_fracs = [s/sum(data_sizes) for s in data_sizes]

        # Create loaders
        train_loaders = [DataLoader(ClientDataset(train_subset, idxs), 
                                   batch_size=args.batch_size, shuffle=True)
                        for idxs in clients_idx]

        # Initialize
        device = args.device
        global_model = model_fn(num_classes=10)
        server = FedAvgServer(global_model, data_fracs, device)

        # Tracking
        train_acc_list, val_acc_list, test_acc_list = [], [], []
        best_val_acc = 0
        patience_counter = 0
        
        # Create output directory
        out_dir = os.path.join(args.save_dir, f"{name}_{p_name}")
        os.makedirs(out_dir, exist_ok=True)

        # Plot data distribution
        plt.figure(figsize=(10, 6))
        plt.bar(range(args.num_clients), data_sizes)
        plt.title(f"{name}-{p_name} Data Distribution")
        plt.xlabel("Client ID")
        plt.ylabel("Number of Samples")
        plt.savefig(os.path.join(out_dir, 'data_dist.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Training rounds
        for r in range(1, args.rounds + 1):
            # Client selection (có thể random subset)
            selected_clients = range(len(train_loaders))  # Chọn tất cả clients
            
            client_models = []
            client_losses = []
            
            # Local training
            for client_id in selected_clients:
                loader = train_loaders[client_id]
                local_model = deepcopy(server.global_model).to(device)
                
                # Adaptive learning rate
                current_lr = args.lr * (0.99 ** (r // 10))
                opt = optim.SGD(local_model.parameters(), lr=current_lr, momentum=0.9)
                
                local_model.train()
                epoch_losses = []
                
                for epoch in range(args.local_epochs):
                    batch_losses = []
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad()
                        loss = F.cross_entropy(local_model(x), y)
                        loss.backward()
                        opt.step()
                        batch_losses.append(loss.item())
                    epoch_losses.append(np.mean(batch_losses))
                
                client_models.append(local_model.cpu())  # Move to CPU to save GPU memory
                client_losses.append(np.mean(epoch_losses))

            # Aggregate
            server.aggregate([model.to(device) for model in client_models])
            
            # Clear client models to free memory
            del client_models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Evaluate
            if r % args.eval_freq == 0 or r == args.rounds:
                val_acc = evaluate(server.global_model, val_loader, device)
                test_acc = evaluate(server.global_model, test_loader, device)
                
                val_acc_list.append(val_acc)
                test_acc_list.append(test_acc)
                
                avg_client_loss = np.mean(client_losses)
                print(f"Round {r:3d}: Val={val_acc*100:5.2f}%, Test={test_acc*100:5.2f}%, Loss={avg_client_loss:.4f}")
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    torch.save(server.global_model.state_dict(), 
                              os.path.join(out_dir, 'model_best.pth'))
                else:
                    patience_counter += 1
                    
                if patience_counter >= args.patience and r > args.rounds // 2:
                    print(f"Early stopping at round {r}")
                    break

        # Save final model and training history
        torch.save(server.global_model.state_dict(), 
                  os.path.join(out_dir, 'model_final.pth'))
        
        # Save training history
        history = {
            'val_acc': val_acc_list,
            'test_acc': test_acc_list,
            'eval_rounds': list(range(args.eval_freq, len(val_acc_list) * args.eval_freq + 1, args.eval_freq))
        }
        with open(os.path.join(out_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        # Plot results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        eval_rounds = range(args.eval_freq, len(val_acc_list) * args.eval_freq + 1, args.eval_freq)
        plt.plot(eval_rounds, [a*100 for a in val_acc_list], 'b-o', label='Validation', markersize=4)
        plt.plot(eval_rounds, [a*100 for a in test_acc_list], 'r-s', label='Test', markersize=4)
        plt.title(f'{name}-{p_name} Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(data_sizes, bins=min(20, args.num_clients), alpha=0.7, edgecolor='black')
        plt.title('Data Size Distribution')
        plt.xlabel('Samples per Client')
        plt.ylabel('Number of Clients')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'results.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store results
        results[p_name] = {
            'best_val_acc': best_val_acc,
            'final_test_acc': test_acc_list[-1] if test_acc_list else 0,
            'data_sizes': data_sizes,
            'training_rounds': len(val_acc_list) * args.eval_freq
        }
        
        print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
        print(f"Final test accuracy: {test_acc_list[-1]*100:.2f}%")
    
    return results

# ---------- Main Execution ----------
if __name__ == '__main__':
    class Args: pass
    args = Args()
    args.seed = 42
    args.data_path = './data'
    args.save_dir = './results'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Federated settings
    args.num_clients = 10
    args.local_epochs = 3
    args.batch_size = 32
    args.rounds = 100
    args.eval_freq = 5
    args.lr = 0.01
    
    # Early stopping
    args.patience = 10

    # Partition settings
    args.alpha = 0.5
    args.main_ratio = 0.6
    args.labels_per_client = 2

    datasets_to_run = [
        ('MNIST', datasets.MNIST),
        ('FashionMNIST', datasets.FashionMNIST)
    ]

    set_seed(args.seed)
    all_results = {}
    
    # Create main results directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    for name, ds_cls in datasets_to_run:
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print(f"{'='*50}")
        results = train_federated_improved(name, ds_cls, args)
        all_results[name] = results
        gc.collect()
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    with open(os.path.join(args.save_dir, f'all_results_{timestamp}.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save as pickle for complete data preservation
    with open(os.path.join(args.save_dir, f'all_results_{timestamp}.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    # Create summary DataFrame and save as CSV
    summary_data = []
    for dataset_name, dataset_results in all_results.items():
        for partition_name, metrics in dataset_results.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Partition': partition_name,
                'Best_Val_Acc': metrics['best_val_acc'],
                'Final_Test_Acc': metrics['final_test_acc'],
                'Training_Rounds': metrics.get('training_rounds', 0),
                'Min_Data_Size': min(metrics['data_sizes']),
                'Max_Data_Size': max(metrics['data_sizes']),
                'Avg_Data_Size': np.mean(metrics['data_sizes'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(args.save_dir, f'summary_{timestamp}.csv'), index=False)
    
    # Summary results
    print(f"\n{'='*50}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*50}")
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        for partition_name, metrics in dataset_results.items():
            print(f"  {partition_name:12s}: Val={metrics['best_val_acc']*100:5.2f}%, Test={metrics['final_test_acc']*100:5.2f}%")
    
    print(f"\nResults saved to: {args.save_dir}")
    print(f"Summary CSV: summary_{timestamp}.csv")
    print(f"Complete results: all_results_{timestamp}.json")
    print("\nAll experiments completed!")