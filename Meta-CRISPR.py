import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np


# ================================
# 1. 图注意力网络 (GAT) 定义
# ================================
class GATModel(nn.Module):
    def __init__(self, node_dim=15, hidden_dim=64, heads=4):
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden_dim, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x.mean(dim=0)  # 全局平均池化得到图嵌入


# ================================
# 2. 特征拼接与MLP模型
# ================================
class MetaMLP(nn.Module):
    def __init__(self, gat_dim=64, species_dim=32, hidden_dim=128):
        super().__init__()
        self.gat = GATModel()
        self.mlp = nn.Sequential(
            nn.Linear(gat_dim + species_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 加载预训练物种嵌入
        self.species_embeddings = pd.read_csv("species_embeddings.csv", index_col=0)

    def forward(self, data):
        # GAT处理图数据
        graph_emb = self.gat(data)  # [64]

        # 获取物种嵌入
        species = data.species
        species_emb = torch.tensor(
            self.species_embeddings.loc[species].values,
            dtype=torch.float32
        )  # [32]

        # 特征拼接
        combined = torch.cat([graph_emb, species_emb], dim=-1)  # [64+32=96]

        # MLP预测
        return self.mlp(combined).sigmoid()


# ================================
# 3. 自定义数据集
# ================================
class CRISPRDataset(Dataset):
    def __init__(self, graph_data_path, species_mapping):
        super().__init__()
        self.graphs = torch.load(graph_data_path)  # 预处理的图数据列表
        self.species_map = pd.read_csv(species_mapping)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        species = self.species_map.iloc[idx]['species']
        data.species = species
        data.y = torch.tensor([self.species_map.iloc[idx]['activity']], dtype=torch.float)
        return data


# ================================
# 4. 训练流程
# ================================
def train():
    # 超参数
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 1e-4

    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MetaMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    # 加载数据
    dataset = CRISPRDataset(
        graph_data_path="graphs.pt",
        species_mapping="species_labels.csv"
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch)
            loss = criterion(pred, batch.y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证集评估
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")


# ================================
# 5. 零样本预测 (Fine-tuning)
# ================================
def fine_tune(species, fewshot_data):
    # 加载基础模型
    model = MetaMLP().eval()

    # 冻结GAT参数
    for param in model.gat.parameters():
        param.requires_grad = False

    # 仅训练MLP头部
    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=1e-5)

    # 进化邻居检索
    species_emb = model.species_embeddings.loc[species].values
    distances = np.linalg.norm(
        model.species_embeddings.values - species_emb,
        axis=1
    )
    neighbor_indices = np.where(distances <= 0.5)[0]

    # 小样本训练
    for _ in range(50):  # 最大50 epoch
        for data in fewshot_data:
            pred = model(data)
            loss = criterion(pred, data.y)
            loss.backward()
            optimizer.step()

    return model