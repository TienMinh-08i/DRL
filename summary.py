#!/usr/bin/env python3
"""
federated_analysis.py - Phân tích và so sánh kết quả Federated Learning

Script này sẽ:
1. Đọc tất cả file metrics.csv từ thư mục kết quả
2. Vẽ biểu đồ so sánh quá trình training giữa các thuật toán
3. Tạo bảng so sánh kết quả tốt nhất
4. Lưu tất cả kết quả dưới dạng ảnh
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Cấu hình matplotlib để hỗ trợ tiếng Việt
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class FederatedAnalyzer:
    def __init__(self, results_root="./results_v2"):
        self.results_root = results_root
        self.algorithms = ["FedAvg", "FedRL-PPO", "FedRL-AC"]
        self.colors = {
            "FedAvg": "#1f77b4",
            "FedRL-PPO": "#ff7f0e", 
            "FedRL-AC": "#2ca02c"
        }
        self.data = defaultdict(lambda: defaultdict(dict))
        self.summary = defaultdict(lambda: defaultdict(dict))
        
    def load_all_metrics(self):
        """Đọc tất cả file metrics.csv từ thư mục kết quả"""
        print("Đang đọc dữ liệu từ các file metrics.csv...")
        
        if not os.path.exists(self.results_root):
            print(f"Thư mục {self.results_root} không tồn tại!")
            return
            
        # Duyệt qua tất cả dataset
        for dataset in os.listdir(self.results_root):
            dataset_path = os.path.join(self.results_root, dataset)
            if not os.path.isdir(dataset_path):
                continue
                
            print(f"\n--- Dataset: {dataset} ---")
            
            # Duyệt qua tất cả partition
            for partition in os.listdir(dataset_path):
                partition_path = os.path.join(dataset_path, partition)
                if not os.path.isdir(partition_path):
                    continue
                    
                print(f"  Partition: {partition}")
                
                # Duyệt qua tất cả thuật toán
                for algorithm in self.algorithms:
                    algo_path = os.path.join(partition_path, algorithm)
                    metrics_file = os.path.join(algo_path, "metrics.csv")
                    
                    if os.path.exists(metrics_file):
                        try:
                            df = pd.read_csv(metrics_file)
                            self.data[dataset][partition][algorithm] = df
                            print(f"    ✓ {algorithm}: {len(df)} rounds")
                        except Exception as e:
                            print(f"    ✗ {algorithm}: Lỗi đọc file - {e}")
                    else:
                        print(f"    ✗ {algorithm}: File không tồn tại")
    
    def calculate_summary(self):
        """Tính toán các chỉ số tóm tắt"""
        print("\nTính toán chỉ số tóm tắt...")
        
        for dataset in self.data:
            for partition in self.data[dataset]:
                for algorithm in self.data[dataset][partition]:
                    df = self.data[dataset][partition][algorithm]
                    
                    # Tính các chỉ số cơ bản
                    max_acc = df['accuracy'].max()
                    final_acc = df['accuracy'].iloc[-1]
                    rounds_to_converge = len(df)
                    
                    # Tính convergence (round đạt 95% accuracy tối đa)
                    target_acc = max_acc * 0.95
                    converge_round = df[df['accuracy'] >= target_acc]['round'].iloc[0] if len(df[df['accuracy'] >= target_acc]) > 0 else rounds_to_converge
                    
                    # Tính độ ổn định (standard deviation của 10 round cuối)
                    stability = df['accuracy'].tail(10).std()
                    
                    self.summary[dataset][partition][algorithm] = {
                        'max_accuracy': max_acc,
                        'final_accuracy': final_acc,
                        'rounds_to_converge': converge_round,
                        'stability': stability,
                        'total_rounds': rounds_to_converge
                    }
    
    def plot_training_curves(self):
        """Vẽ biểu đồ quá trình training"""
        print("\nVẽ biểu đồ quá trình training...")
        
        for dataset in self.data:
            for partition in self.data[dataset]:
                # Tạo figure với subplots
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'Training Progress - {dataset} - {partition}', fontsize=16, fontweight='bold')
                
                # 1. Accuracy curves
                ax1 = axes[0, 0]
                for algorithm in self.algorithms:
                    if algorithm in self.data[dataset][partition]:
                        df = self.data[dataset][partition][algorithm]
                        ax1.plot(df['round'], df['accuracy'], 
                                label=algorithm, color=self.colors[algorithm], linewidth=2)
                
                ax1.set_xlabel('Round')
                ax1.set_ylabel('Test Accuracy')
                ax1.set_title('Accuracy Comparison')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 2. Reward curves (chỉ cho FedRL algorithms)
                ax2 = axes[0, 1]
                for algorithm in ["FedRL-PPO", "FedRL-AC"]:
                    if algorithm in self.data[dataset][partition]:
                        df = self.data[dataset][partition][algorithm]
                        if 'reward' in df.columns:
                            ax2.plot(df['round'], df['reward'], 
                                    label=algorithm, color=self.colors[algorithm], linewidth=2)
                
                ax2.set_xlabel('Round')
                ax2.set_ylabel('Reward')
                ax2.set_title('Reward Comparison (FedRL methods)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 3. Convergence comparison
                ax3 = axes[1, 0]
                algorithms_present = [alg for alg in self.algorithms if alg in self.data[dataset][partition]]
                convergence_rounds = [self.summary[dataset][partition][alg]['rounds_to_converge'] 
                                    for alg in algorithms_present]
                
                bars = ax3.bar(algorithms_present, convergence_rounds, 
                              color=[self.colors[alg] for alg in algorithms_present])
                ax3.set_ylabel('Rounds to Converge')
                ax3.set_title('Convergence Speed')
                ax3.tick_params(axis='x', rotation=45)
                
                # Thêm giá trị trên các cột
                for bar, value in zip(bars, convergence_rounds):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{value}', ha='center', va='bottom')
                
                # 4. Final accuracy comparison
                ax4 = axes[1, 1]
                final_accs = [self.summary[dataset][partition][alg]['final_accuracy'] 
                             for alg in algorithms_present]
                
                bars = ax4.bar(algorithms_present, final_accs, 
                              color=[self.colors[alg] for alg in algorithms_present])
                ax4.set_ylabel('Final Accuracy')
                ax4.set_title('Final Accuracy Comparison')
                ax4.tick_params(axis='x', rotation=45)
                ax4.set_ylim(0, 1)
                
                # Thêm giá trị trên các cột
                for bar, value in zip(bars, final_accs):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Tạo thư mục và lưu ảnh
                save_dir = os.path.join(self.results_root, dataset, partition)
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, 'training_comparison.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ Saved: {dataset}/{partition}/training_comparison.png")
    
    def create_summary_table(self):
        """Tạo bảng tóm tắt kết quả"""
        print("\nTạo bảng tóm tắt kết quả...")
        
        for dataset in self.summary:
            # Tạo DataFrame cho bảng tóm tắt
            rows = []
            for partition in self.summary[dataset]:
                for algorithm in self.summary[dataset][partition]:
                    metrics = self.summary[dataset][partition][algorithm]
                    rows.append({
                        'Dataset': dataset,
                        'Partition': partition,
                        'Algorithm': algorithm,
                        'Max Accuracy': f"{metrics['max_accuracy']:.4f}",
                        'Final Accuracy': f"{metrics['final_accuracy']:.4f}",
                        'Convergence Round': metrics['rounds_to_converge'],
                        'Stability (std)': f"{metrics['stability']:.4f}",
                        'Total Rounds': metrics['total_rounds']
                    })
            
            df_summary = pd.DataFrame(rows)
            
            # Tạo heatmap cho Max Accuracy
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Summary Report - {dataset}', fontsize=16, fontweight='bold')
            
            # 1. Max Accuracy Heatmap
            pivot_max = df_summary.pivot(index='Partition', columns='Algorithm', values='Max Accuracy')
            pivot_max = pivot_max.astype(float)
            
            sns.heatmap(pivot_max, annot=True, fmt='.4f', cmap='YlOrRd', 
                       ax=axes[0, 0], cbar_kws={'label': 'Max Accuracy'})
            axes[0, 0].set_title('Maximum Accuracy Achieved')
            
            # 2. Final Accuracy Heatmap
            pivot_final = df_summary.pivot(index='Partition', columns='Algorithm', values='Final Accuracy')
            pivot_final = pivot_final.astype(float)
            
            sns.heatmap(pivot_final, annot=True, fmt='.4f', cmap='YlOrRd',
                       ax=axes[0, 1], cbar_kws={'label': 'Final Accuracy'})
            axes[0, 1].set_title('Final Accuracy')
            
            # 3. Convergence Speed Heatmap
            pivot_conv = df_summary.pivot(index='Partition', columns='Algorithm', values='Convergence Round')
            pivot_conv = pivot_conv.astype(float)
            
            sns.heatmap(pivot_conv, annot=True, fmt='.0f', cmap='YlOrRd_r',
                       ax=axes[1, 0], cbar_kws={'label': 'Rounds'})
            axes[1, 0].set_title('Convergence Speed (Lower is Better)')
            
            # 4. Stability Heatmap
            pivot_stab = df_summary.pivot(index='Partition', columns='Algorithm', values='Stability (std)')
            pivot_stab = pivot_stab.astype(float)
            
            sns.heatmap(pivot_stab, annot=True, fmt='.4f', cmap='YlOrRd_r',
                       ax=axes[1, 1], cbar_kws={'label': 'Std Dev'})
            axes[1, 1].set_title('Stability (Lower is Better)')
            
            plt.tight_layout()
            
            # Lưu ảnh
            save_dir = os.path.join(self.results_root, dataset)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'summary_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Lưu bảng CSV
            df_summary.to_csv(os.path.join(save_dir, 'summary_table.csv'), index=False)
            
            print(f"  ✓ Saved: {dataset}/summary_heatmap.png")
            print(f"  ✓ Saved: {dataset}/summary_table.csv")
    
    def create_overall_comparison(self):
        """Tạo biểu đồ so sánh tổng quan"""
        print("\nTạo biểu đồ so sánh tổng quan...")
        
        # Tập hợp tất cả dữ liệu
        all_results = []
        for dataset in self.summary:
            for partition in self.summary[dataset]:
                for algorithm in self.summary[dataset][partition]:
                    metrics = self.summary[dataset][partition][algorithm]
                    all_results.append({
                        'Dataset': dataset,
                        'Partition': partition,
                        'Algorithm': algorithm,
                        'Max_Accuracy': metrics['max_accuracy'],
                        'Final_Accuracy': metrics['final_accuracy'],
                        'Convergence_Round': metrics['rounds_to_converge'],
                        'Stability': metrics['stability']
                    })
        
        df_all = pd.DataFrame(all_results)
        
        # Tạo figure với multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Overall Comparison Across All Experiments', fontsize=16, fontweight='bold')
        
        # 1. Box plot cho Max Accuracy
        df_all.boxplot(column='Max_Accuracy', by='Algorithm', ax=axes[0, 0])
        axes[0, 0].set_title('Max Accuracy Distribution')
        axes[0, 0].set_xlabel('Algorithm')
        axes[0, 0].set_ylabel('Max Accuracy')
        
        # 2. Box plot cho Convergence Round
        df_all.boxplot(column='Convergence_Round', by='Algorithm', ax=axes[0, 1])
        axes[0, 1].set_title('Convergence Speed Distribution')
        axes[0, 1].set_xlabel('Algorithm')
        axes[0, 1].set_ylabel('Rounds to Converge')
        
        # 3. Scatter plot: Accuracy vs Convergence
        for algorithm in self.algorithms:
            if algorithm in df_all['Algorithm'].values:
                subset = df_all[df_all['Algorithm'] == algorithm]
                axes[1, 0].scatter(subset['Convergence_Round'], subset['Max_Accuracy'], 
                                  label=algorithm, color=self.colors[algorithm], 
                                  alpha=0.7, s=50)
        
        axes[1, 0].set_xlabel('Rounds to Converge')
        axes[1, 0].set_ylabel('Max Accuracy')
        axes[1, 0].set_title('Accuracy vs Convergence Speed')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Average performance by algorithm
        avg_metrics = df_all.groupby('Algorithm').agg({
            'Max_Accuracy': 'mean',
            'Final_Accuracy': 'mean',
            'Convergence_Round': 'mean',
            'Stability': 'mean'
        }).round(4)
        
        # Tạo bar chart cho average performance
        x = np.arange(len(avg_metrics.index))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, avg_metrics['Max_Accuracy'], width, 
                      label='Max Accuracy', alpha=0.8)
        axes[1, 1].bar(x + width/2, avg_metrics['Final_Accuracy'], width, 
                      label='Final Accuracy', alpha=0.8)
        
        axes[1, 1].set_xlabel('Algorithm')
        axes[1, 1].set_ylabel('Average Accuracy')
        axes[1, 1].set_title('Average Performance Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(avg_metrics.index)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Lưu ảnh
        plt.savefig(os.path.join(self.results_root, 'overall_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Lưu bảng thống kê tổng quan
        avg_metrics.to_csv(os.path.join(self.results_root, 'overall_statistics.csv'))
        
        print(f"  ✓ Saved: overall_comparison.png")
        print(f"  ✓ Saved: overall_statistics.csv")
    
    def find_best_configurations(self):
        """Tìm cấu hình tốt nhất cho mỗi dataset"""
        print("\nTìm cấu hình tốt nhất...")
        
        best_configs = {}
        
        for dataset in self.summary:
            best_configs[dataset] = {
                'max_accuracy': {'config': None, 'value': 0},
                'fastest_convergence': {'config': None, 'value': float('inf')},
                'most_stable': {'config': None, 'value': float('inf')}
            }
            
            for partition in self.summary[dataset]:
                for algorithm in self.summary[dataset][partition]:
                    metrics = self.summary[dataset][partition][algorithm]
                    config = f"{partition}_{algorithm}"
                    
                    # Best accuracy
                    if metrics['max_accuracy'] > best_configs[dataset]['max_accuracy']['value']:
                        best_configs[dataset]['max_accuracy'] = {
                            'config': config,
                            'value': metrics['max_accuracy']
                        }
                    
                    # Fastest convergence
                    if metrics['rounds_to_converge'] < best_configs[dataset]['fastest_convergence']['value']:
                        best_configs[dataset]['fastest_convergence'] = {
                            'config': config,
                            'value': metrics['rounds_to_converge']
                        }
                    
                    # Most stable
                    if metrics['stability'] < best_configs[dataset]['most_stable']['value']:
                        best_configs[dataset]['most_stable'] = {
                            'config': config,
                            'value': metrics['stability']
                        }
        
        # Tạo báo cáo
        report = []
        for dataset in best_configs:
            report.append({
                'Dataset': dataset,
                'Best_Accuracy_Config': best_configs[dataset]['max_accuracy']['config'],
                'Best_Accuracy_Value': f"{best_configs[dataset]['max_accuracy']['value']:.4f}",
                'Fastest_Config': best_configs[dataset]['fastest_convergence']['config'],
                'Fastest_Rounds': best_configs[dataset]['fastest_convergence']['value'],
                'Most_Stable_Config': best_configs[dataset]['most_stable']['config'],
                'Stability_Value': f"{best_configs[dataset]['most_stable']['value']:.4f}"
            })
        
        df_best = pd.DataFrame(report)
        df_best.to_csv(os.path.join(self.results_root, 'best_configurations.csv'), index=False)
        
        print(f"  ✓ Saved: best_configurations.csv")
        print("\nCấu hình tốt nhất:")
        print(df_best.to_string(index=False))
    
    def run_analysis(self):
        """Chạy toàn bộ phân tích"""
        print("=== BẮT ĐẦU PHÂN TÍCH FEDERATED LEARNING ===")
        
        # Đọc dữ liệu
        self.load_all_metrics()
        
        # Kiểm tra xem có dữ liệu không
        if not self.data:
            print("Không tìm thấy dữ liệu nào để phân tích!")
            return
        
        # Tính toán các chỉ số
        self.calculate_summary()
        
        # Tạo các biểu đồ và báo cáo
        self.plot_training_curves()
        self.create_summary_table()
        self.create_overall_comparison()
        self.find_best_configurations()
        
        print("\n=== HOÀN THÀNH PHÂN TÍCH ===")
        print(f"Tất cả kết quả đã được lưu trong thư mục: {self.results_root}")


def main():
    """Hàm main để chạy phân tích"""
    # Khởi tạo analyzer
    analyzer = FederatedAnalyzer(results_root="./results_100clients_100round")
    
    # Chạy phân tích
    analyzer.run_analysis()


if __name__ == "__main__":
    main()