import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import os
from torch.utils.data import Dataset, DataLoader
import random


class RNADNAInteractionEncoder(nn.Module):
    """基于图注意力网络的sgRNA-DNA相互作用编码器"""

    def __init__(self, sgRNA_length=23, dna_length=23, num_features=4,
                 hidden_channels=64, num_heads=4, dropout=0.2):
        super().__init__()

        # 验证初始化参数
        if sgRNA_length != 23 or dna_length != 23:
            raise ValueError("sgRNA_length和dna_length必须为23")


        # 序列独热编码参数
        self.sgRNA_length = sgRNA_length
        self.dna_length = dna_length
        self.num_features = num_features  # A, T, C, G

        # 图注意力层
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout)

        # 用于处理物种进化距离的MLP
        self.evo_mlp = nn.Sequential(
            nn.Linear(1, hidden_channels),  # 假设进化距离是标量
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

    def encode_sequence(self, sequence):
        """将DNA/RNA序列转换为独热编码"""
        encoder = OneHotEncoder(categories=[['A', 'T', 'C', 'G']], sparse_output=False)
        encoded = encoder.fit_transform(np.array(sequence).reshape(-1, 1))
        return torch.tensor(encoded, dtype=torch.float32)

    def build_graph(self, sgRNA, dna):
        """基于Watson-Crick配对构建图结构"""

        # 验证输入序列长度
        if len(sgRNA) != self.sgRNA_length or len(dna) != self.dna_length:
            raise ValueError(f"输入序列长度必须为{self.sgRNA_length}bp，"
                             f"但得到sgRNA长度: {len(sgRNA)}, DNA长度: {len(dna)}")

        # 节点特征：独热编码的序列
        sgRNA_nodes = self.encode_sequence(sgRNA)
        dna_nodes = self.encode_sequence(dna)
        nodes = torch.cat([sgRNA_nodes, dna_nodes], dim=0)

        # 构建边：基于碱基配对规则 (A-T, C-G)
        edges = []
        for i in range(self.sgRNA_length):
            for j in range(self.dna_length):
                # 简化的碱基配对规则
                if (sgRNA[i] == 'A' and dna[j] == 'T') or \
                        (sgRNA[i] == 'T' and dna[j] == 'A') or \
                        (sgRNA[i] == 'C' and dna[j] == 'G') or \
                        (sgRNA[i] == 'G' and dna[j] == 'C'):
                    # sgRNA到DNA的边
                    edges.append([i, self.sgRNA_length + j])
                    # DNA到sgRNA的边 (双向)
                    edges.append([self.sgRNA_length + j, i])

        # 转换为PyTorch Geometric的边格式
        if len(edges) == 0:
            # 如果没有边，创建自环以避免计算问题
            edge_index = torch.arange(nodes.size(0)).repeat(2, 1)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # 节点所属的图（batch信息）
        batch = torch.zeros(nodes.size(0), dtype=torch.long)

        return nodes, edge_index, batch

    def forward(self, sgRNA, dna, evolutionary_distance):
        # 验证序列长度
        if len(sgRNA) != len(dna):
            raise ValueError("sgRNA和DNA序列长度必须相同")
        """模型前向传播"""
        # 构建图
        x, edge_index, batch = self.build_graph(sgRNA, dna)

        # 图注意力编码
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)

        # 全局池化
        graph_embedding = global_mean_pool(x, batch)

        # 处理进化距离
        evo_embedding = self.evo_mlp(evolutionary_distance.view(-1, 1))

        # 特征拼接
        combined = torch.cat([graph_embedding, evo_embedding], dim=1)
        fused_embedding = self.fusion(combined)

        return fused_embedding


class CombinedModel(nn.Module):
    """整合序列编码和进化距离的完整模型"""

    def __init__(self, sgRNA_length=23, dna_length=23, num_classes=2,
                 hidden_channels=64, num_heads=4, dropout=0.2):
        super().__init__()

        # 序列编码器
        self.sequence_encoder = RNADNAInteractionEncoder(
            sgRNA_length=sgRNA_length,
            dna_length=dna_length,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            dropout=dropout
        )

        # MLP分类器
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, sgRNA, dna, evolutionary_distance):
        """模型前向传播"""
        # 获取融合的特征表示
        features = self.sequence_encoder(sgRNA, dna, evolutionary_distance)

        # 通过MLP进行预测
        logits = self.mlp(features)
        return logits


class InteractionDataset(Dataset):
    """sgRNA-DNA相互作用数据集"""

    def __init__(self, data, labels=None, is_test=False):
        self.data = data
        self.labels = labels
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 直接提取序列（确保是长度为23的碱基列表）
        sgRNA = item['sgRNA']
        dna = item['dna']

        # 验证序列长度
        if len(sgRNA) != 23 or len(dna) != 23:
            print(f"数据集中样本 {idx} 的序列长度错误: sgRNA={len(sgRNA)}, DNA={len(dna)}")
            print(f"sgRNA内容: {sgRNA}")
            print(f"DNA内容: {dna}")
            raise ValueError(f"序列长度必须为23bp")

        evolutionary_distance = torch.tensor(item['evolutionary_distance'], dtype=torch.float32)

        if self.is_test:
            return sgRNA, dna, evolutionary_distance
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return sgRNA, dna, evolutionary_distance, label


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=10):
    """训练模型并进行验证"""
    best_val_loss = float('inf')
    best_model = None
    early_stopping_counter = 0


    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            if len(batch) == 4:  # 训练数据包含标签
                sgRNA_batch, dna_batch, evo_dist_batch, labels = batch
                labels = labels.to(device)
            else:  # 测试数据不包含标签
                sgRNA_batch, dna_batch, evo_dist_batch = batch
                labels = None

            optimizer.zero_grad()

            # 前向传播 - 逐个处理样本
            batch_outputs = []
            for i in range(len(sgRNA_batch)):
                # 获取单个样本
                sgRNA = sgRNA_batch[i]
                dna = dna_batch[i]
                evo_dist = evo_dist_batch[i].to(device)

                # 验证单个序列长度
                if len(sgRNA) != 23 or len(dna) != 23:
                    print(f"批次中样本 {i} 的序列长度错误: sgRNA={len(sgRNA)}, DNA={len(dna)}")
                    print(f"sgRNA内容: {sgRNA}")
                    print(f"DNA内容: {dna}")
                    raise ValueError(f"序列长度必须为23bp")

                # 处理单个样本
                output = model(sgRNA, dna, evo_dist)
                batch_outputs.append(output)

            # 合并所有样本的输出
            outputs = torch.cat(batch_outputs, dim=0)

    # 加载最佳模型
    model.load_state_dict(best_model)
    return model


def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 4:  # 训练数据包含标签
                sgRNA, dna, evo_dist, labels = batch
                labels = labels.to(device)
                all_labels.extend(labels.cpu().numpy())
            else:  # 测试数据不包含标签
                sgRNA, dna, evo_dist = batch

            # 前向传播
            outputs = []
            for i in range(len(sgRNA)):
                output = model(sgRNA[i], dna[i], evo_dist[i].to(device))
                outputs.append(output)

            outputs = torch.cat(outputs, dim=0)
            probs = F.softmax(outputs, dim=1)

            # 计算预测结果
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算评估指标
    results = {}
    if len(all_labels) > 0:  # 只有当有标签时才能计算评估指标
        results['accuracy'] = accuracy_score(all_labels, all_preds)
        if len(set(all_labels)) == 2:  # 二分类问题
            results['auc'] = roc_auc_score(all_labels, [p[1] for p in all_probs])

    return results, all_preds, all_probs


def save_model(model, path):
    """保存模型为pkl文件"""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存至 {path}")


def load_model(path):
    """从pkl文件加载模型"""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"模型已从 {path} 加载")
    return model


def generate_sample_data(num_samples=1000, seq_length=23):
    """生成示例数据集，强制序列长度为23bp"""
    nucleotides = ['A', 'T', 'C', 'G']
    data = []
    labels = []

    for i in range(num_samples):
        # 强制序列长度为23bp
        seq_length = 23

        # 生成随机sgRNA序列
        sgRNA = [random.choice(nucleotides) for _ in range(seq_length)]

        # 生成DNA序列：前半部分互补，后半部分随机
        dna_complementary = [{'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}[base] for base in sgRNA[:seq_length // 2]]
        dna_random = [random.choice(nucleotides) for _ in range(seq_length - len(dna_complementary))]
        dna = dna_complementary + dna_random

        # 严格验证序列长度和类型
        if not isinstance(sgRNA, list) or len(sgRNA) != 23:
            raise TypeError(f"sgRNA必须是长度为23的列表，但得到: {type(sgRNA)}, 长度={len(sgRNA)}")
        if not all(isinstance(base, str) for base in sgRNA):
            raise TypeError(f"sgRNA的每个元素必须是字符串，但包含: {[type(b) for b in sgRNA]}")

        if not isinstance(dna, list) or len(dna) != 23:
            raise TypeError(f"DNA必须是长度为23的列表，但包含: {type(dna)}, 长度={len(dna)}")
        if not all(isinstance(base, str) for base in dna):
            raise TypeError(f"DNA的每个元素必须是字符串，但包含: {[type(b) for b in dna]}")

        # 生成进化距离
        evolutionary_distance = random.uniform(0.01, 0.99)

        # 生成标签
        match_ratio = sum(1 for b1, b2 in zip(sgRNA, dna)
                          if (b1 == 'A' and b2 == 'T') or
                          (b1 == 'T' and b2 == 'A') or
                          (b1 == 'C' and b2 == 'G') or
                          (b1 == 'G' and b2 == 'C')) / seq_length
        label = 1 if match_ratio > 0.5 + random.uniform(-0.2, 0.2) else 0

        data.append({
            'sgRNA': sgRNA,
            'dna': dna,
            'evolutionary_distance': evolutionary_distance
        })
        labels.append(label)

    print(f"成功生成 {num_samples} 条样本数据，序列长度均为 23bp")
    return data, labels


def main():
    """主函数：训练、评估并保存模型"""
    # 设置随机种子以确保结果可复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 生成示例数据
    print("生成示例数据...")
    data, labels = generate_sample_data(num_samples=1000)

    # 划分训练集和测试集
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    # 创建数据加载器
    train_dataset = InteractionDataset(train_data, train_labels)
    test_dataset = InteractionDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset,  shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    for batch_idx, batch in enumerate(train_loader):
        print(f"\n批次 {batch_idx} 的结构:")
        print(f"  batch 类型: {type(batch)}, 长度: {len(batch)}")

        # 解包批次
        sgRNA_batch, dna_batch, evo_dist_batch, labels = batch

        # 检查每个组件的类型和形状
        print(f"  sgRNA_batch 类型: {type(sgRNA_batch)}, 长度: {len(sgRNA_batch)}")
        print(f"  dna_batch 类型: {type(dna_batch)}, 长度: {len(dna_batch)}")
        print(f"  evo_dist_batch 类型: {type(evo_dist_batch)}, 形状: {evo_dist_batch.shape}")
        print(f"  labels 类型: {type(labels)}, 形状: {labels.shape}")

        # 检查第一个样本的序列长度
        if len(sgRNA_batch) > 0:
            print(f"  第一个 sgRNA 样本: 类型={type(sgRNA_batch[0])}, 长度={len(sgRNA_batch[0])}")
            print(f"  内容: {sgRNA_batch[0][:10]}...")  # 打印前10个碱基

        break  # 只检查第一个批次

    # 初始化模型
    model = CombinedModel().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("开始训练模型...")
    model = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=50)

    # 评估模型
    print("评估模型性能...")
    results, _, _ = evaluate_model(model, test_loader, device)
    print(f"测试集准确率: {results['accuracy']:.4f}")
    if 'auc' in results:
        print(f"测试集AUC: {results['auc']:.4f}")

    # 保存模型
    model_path = 'sgRNA_DNA_interaction_model.pkl'
    save_model(model, model_path)

    # 示例：加载模型并进行预测
    loaded_model = load_model(model_path)
    loaded_model.to(device)

    # 对单个样本进行预测
    sample = test_data[0]
    sgRNA, dna, evo_dist = sample['sgRNA'], sample['dna'], sample['evolutionary_distance']

    loaded_model.eval()
    with torch.no_grad():
        output = loaded_model(sgRNA, dna, torch.tensor(evo_dist, dtype=torch.float32).to(device))
        probs = F.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    print(f"示例预测: 类别 {prediction}, 概率分布: {probs.cpu().numpy()}")


if __name__ == "__main__":
    main()