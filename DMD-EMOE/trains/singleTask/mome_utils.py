"""
MoME工具函数：模态权重可视化、分析工具等
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def visualize_modality_weights(weights, labels, save_path=None):
    """
    可视化模态权重分布

    Args:
        weights: (batch_size, num_modalities) 模态权重
        labels: (batch_size,) 样本标签
        save_path: 保存路径，如果为None则不保存
    """
    num_modalities = weights.shape[1]

    fig, axes = plt.subplots(1, num_modalities + 1, figsize=(15, 4))

    # 绘制每个模态的权重分布
    for i in range(num_modalities):
        ax = axes[i]
        modality_weights = weights[:, i].cpu().numpy()

        ax.hist(modality_weights, bins=20, alpha=0.7, color=f'C{i}')
        ax.set_title(f'Modality {i} Weight Distribution')
        ax.set_xlabel('Weight')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    # 绘制权重与标签的关系
    ax = axes[-1]
    unique_labels = torch.unique(labels)

    for label in unique_labels:
        mask = labels == label
        label_weights = weights[mask].mean(dim=0).cpu().numpy()

        x = np.arange(num_modalities)
        ax.plot(x, label_weights, marker='o', label=f'Label {label.item()}')

    ax.set_title('Average Weights per Label')
    ax.set_xlabel('Modality')
    ax.set_ylabel('Average Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def analyze_modality_importance(model, dataloader, device):
    """
    分析模态重要性：计算每个模态的平均权重和预测误差

    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备

    Returns:
        dict: 包含模态重要性分析结果
    """
    model.eval()

    total_weights = []
    modality_errors = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            # 获取数据
            vision = batch['vision'].to(device)
            text = batch['text'].to(device)
            labels = batch['labels']['M'].to(device).view(-1, 1)

            # 前向传播
            output = model(text, torch.zeros_like(vision), vision, is_distill=True)

            if 'mome_weights' in output:
                weights = output['mome_weights'].cpu()
                total_weights.append(weights)

                # 计算模态预测误差
                if 'visual_pred' in output and 'language_pred' in output:
                    visual_error = torch.abs(output['visual_pred'] - labels).mean().cpu()
                    language_error = torch.abs(output['language_pred'] - labels).mean().cpu()
                    modality_errors.append([visual_error, language_error])

            labels_list.append(labels.cpu())

    if len(total_weights) > 0:
        total_weights = torch.cat(total_weights, dim=0)
        modality_errors = torch.tensor(modality_errors) if modality_errors else None
        labels_list = torch.cat(labels_list, dim=0)

        analysis_results = {
            'avg_weights': total_weights.mean(dim=0).numpy(),
            'std_weights': total_weights.std(dim=0).numpy(),
            'min_weights': total_weights.min(dim=0)[0].numpy(),
            'max_weights': total_weights.max(dim=0)[0].numpy()
        }

        if modality_errors is not None:
            analysis_results['avg_errors'] = modality_errors.mean(dim=0).numpy()
            analysis_results['correlation'] = torch.corrcoef(
                torch.stack([total_weights[:, 0], total_weights[:, 1]])
            ).numpy()

        return analysis_results
    else:
        return None


def visualize_feature_space(features, labels, weights, save_path=None):
    """
    可视化特征空间和模态权重

    Args:
        features: 特征列表 [visual_feat, language_feat]
        labels: 样本标签
        weights: 模态权重
        save_path: 保存路径
    """
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)

    # 准备特征
    visual_feat = features[0].cpu().numpy()
    language_feat = features[1].cpu().numpy()
    labels = labels.cpu().numpy()
    weights = weights.cpu().numpy()

    # 降维
    visual_2d = tsne.fit_transform(visual_feat)
    language_2d = tsne.fit_transform(language_feat)

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 视觉特征空间
    ax = axes[0, 0]
    scatter = ax.scatter(visual_2d[:, 0], visual_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.set_title('Visual Feature Space')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax)

    # 2. 语言特征空间
    ax = axes[0, 1]
    scatter = ax.scatter(language_2d[:, 0], language_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.set_title('Language Feature Space')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax)

    # 3. 权重分布散点图
    ax = axes[1, 0]
    scatter = ax.scatter(weights[:, 0], weights[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    ax.set_title('Modality Weight Distribution')
    ax.set_xlabel('Visual Weight')
    ax.set_ylabel('Language Weight')
    ax.axline((0, 0), (1, 1), color='gray', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax)

    # 4. 权重直方图
    ax = axes[1, 1]
    ax.hist(weights[:, 0], bins=20, alpha=0.5, label='Visual', color='blue')
    ax.hist(weights[:, 1], bins=20, alpha=0.5, label='Language', color='red')
    ax.set_title('Weight Histograms')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature visualization saved to {save_path}")

    plt.show()


def compute_modality_contribution(model, dataloader, device, num_samples=100):
    """
    计算每个模态对最终预测的贡献度

    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        num_samples: 采样数量

    Returns:
        dict: 模态贡献度统计
    """
    model.eval()

    contributions = {'visual': [], 'language': []}

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            vision = batch['vision'].to(device)
            text = batch['text'].to(device)
            labels = batch['labels']['M'].to(device).view(-1, 1)

            # 完整前向传播
            output_full = model(text, torch.zeros_like(vision), vision, is_distill=True)
            full_pred = output_full['output_logit']

            # 仅视觉模态
            if 'visual_pred' in output_full:
                visual_pred = output_full['visual_pred']
                visual_contribution = 1.0 - torch.abs(visual_pred - full_pred) / torch.abs(full_pred)
                contributions['visual'].append(visual_contribution.mean().item())

            # 仅语言模态
            if 'language_pred' in output_full:
                language_pred = output_full['language_pred']
                language_contribution = 1.0 - torch.abs(language_pred - full_pred) / torch.abs(full_pred)
                contributions['language'].append(language_contribution.mean().item())

    # 计算统计信息
    stats = {}
    for modality, values in contributions.items():
        if values:
            stats[f'{modality}_mean'] = np.mean(values)
            stats[f'{modality}_std'] = np.std(values)
            stats[f'{modality}_min'] = np.min(values)
            stats[f'{modality}_max'] = np.max(values)

    return stats