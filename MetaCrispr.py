import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
from typing import List, Tuple, Dict, Any
import pickle
import os
import pandas as pd


class SGRNAGraphEncoder(nn.Module):
    """基于图注意力机制的sgRNA:DNA相互作用编码器"""

    def __init__(self, feature_dim: int = 5, output_dim: int = 72):
        super(SGRNAGraphEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim  # 显式设置输出维度
        self.gat_layer = GATConv(feature_dim, 64, heads=1, concat=True)
        self.feature_extractor = None

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        base_to_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
        one_hot = torch.zeros(len(sequence), self.feature_dim)
        for i, base in enumerate(sequence):
            if base in base_to_idx:
                one_hot[i, base_to_idx[base]] = 1
            else:
                one_hot[i, base_to_idx['N']] = 1
        return one_hot

    def build_graph(self, sgRNA: str, dna: str) -> Data:
        x_sgRNA = self.encode_sequence(sgRNA)
        x_dna = self.encode_sequence(dna)
        x = torch.cat([x_sgRNA, x_dna], dim=0)

        edges = []
        sgRNA_len = len(sgRNA)
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}

        for i in range(len(sgRNA)):
            for j in range(len(dna)):
                if complement.get(sgRNA[i], 'N') == dna[j]:
                    edges.append([i, sgRNA_len + j])
                    edges.append([sgRNA_len + j, i])

        # 确保至少有一条边，避免空图问题
        if len(edges) == 0:
            # 创建自环
            edges = [[i, i] for i in range(x.shape[0])]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)

    def forward(self, sgRNA: str, dna: str) -> torch.Tensor:
        graph = self.build_graph(sgRNA, dna)

        # 处理图特征
        x = self.gat_layer(graph.x, graph.edge_index)

        # 使用全局池化获取固定长度表示
        x = torch.mean(x, dim=0, keepdim=True)  # 全局平均池化，得到[1, feature_dim]

        # 动态创建特征提取器
        if self.feature_extractor is None:
            input_dim = x.shape[1]
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, self.output_dim),  # 固定输出维度为72
                nn.ReLU()
            )

        features = self.feature_extractor(x)
        return features


class MetaLearner(nn.Module):
    """元学习器，整合进化距离和图注意力特征"""

    def __init__(self, evolution_distance_dim: int = 72, hidden_dim: int = 64):
        """
        初始化MetaLearner模型

        参数:
            evolution_distance_dim: 进化距离特征的维度，默认72
            hidden_dim: 隐藏层维度，默认64
        """
        super(MetaLearner, self).__init__()

        # 进化特征处理器
        self.evolution_processor = nn.Sequential(
            nn.Linear(evolution_distance_dim, hidden_dim),
            nn.ReLU()
        )

        # 组合特征处理器
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim + 72, 128),  # 明确输入维度为进化特征维度+图特征维度
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.predictor = nn.Linear(32, 1)  # 最终预测器

    def forward(self, evolution_distance: torch.Tensor, graph_features: torch.Tensor) -> torch.Tensor:
        # 确保进化距离特征形状正确
        if evolution_distance.dim() == 1:
            evolution_distance = evolution_distance.unsqueeze(0)  # [72] -> [1, 72]

        # 处理进化特征
        evolution_features = self.evolution_processor(evolution_distance)

        # 拼接特征
        combined_features = torch.cat([evolution_features, graph_features], dim=1)

        # 处理组合特征
        processed_features = self.feature_processor(combined_features)

        # 预测活性
        activity = self.predictor(processed_features)

        return activity


class SGRNAMetaLearningFramework:
    """sgRNA打靶活性预测元学习框架"""

    def __init__(self, learning_rate: float = 0.001, evolution_distance_dim: int = 72):
        self.graph_encoder = SGRNAGraphEncoder(output_dim=72)  # 明确图编码器输出维度
        self.meta_learner = MetaLearner(evolution_distance_dim=evolution_distance_dim)
        self.optimizer = optim.Adam(
            list(self.graph_encoder.parameters()) + list(self.meta_learner.parameters()),
            lr=learning_rate
        )
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_encoder.to(self.device)
        self.meta_learner.to(self.device)

    def train_step(self, batch: List[Dict[str, Any]]) -> float:
        self.graph_encoder.train()
        self.meta_learner.train()
        self.optimizer.zero_grad()
        total_loss = 0.0

        for sample in batch:
            # 确保进化距离是正确的张量
            evolution_distance = torch.tensor(
                sample['evolution_distance'],
                dtype=torch.float32,
                device=self.device
            ).view(1, -1)  # 确保形状为[1, 72]

            sgRNA = sample['sgRNA']
            dna = sample['dna']
            target_activity = torch.tensor([sample['activity']], dtype=torch.float32, device=self.device)

            graph_features = self.graph_encoder(sgRNA, dna).to(self.device)
            predicted_activity = self.meta_learner(evolution_distance, graph_features)

            # 确保目标形状与预测形状匹配
            target_activity = target_activity.view_as(predicted_activity)

            loss = self.criterion(predicted_activity, target_activity)
            total_loss += loss.item()
            loss.backward()

        self.optimizer.step()
        return total_loss / len(batch)

    def fine_tune(self, task_data: List[Dict[str, Any]], num_epochs: int = 5, lr: float = 0.0001) -> None:
        fine_tune_optimizer = optim.Adam(
            list(self.graph_encoder.parameters()) + list(self.meta_learner.parameters()),
            lr=lr
        )

        self.graph_encoder.train()
        self.meta_learner.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            for sample in task_data:
                # 确保进化距离是正确的张量
                evolution_distance = torch.tensor(
                    sample['evolution_distance'],
                    dtype=torch.float32,
                    device=self.device
                ).view(1, -1)  # 确保形状为[1, 72]

                sgRNA = sample['sgRNA']
                dna = sample['dna']
                target_activity = torch.tensor([sample['activity']], dtype=torch.float32, device=self.device)

                graph_features = self.graph_encoder(sgRNA, dna).to(self.device)
                predicted_activity = self.meta_learner(evolution_distance, graph_features)

                # 确保目标形状与预测形状匹配
                target_activity = target_activity.view_as(predicted_activity)

                loss = self.criterion(predicted_activity, target_activity)
                total_loss += loss.item()

                fine_tune_optimizer.zero_grad()
                loss.backward()
                fine_tune_optimizer.step()

            print(f"Fine-tuning Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(task_data):.4f}")

    def predict(self, evolution_distance: float, sgRNA: str, dna: str) -> float:
        self.graph_encoder.eval()
        self.meta_learner.eval()

        with torch.no_grad():
            # 确保进化距离是正确的张量
            evolution_tensor = torch.tensor(
                evolution_distance,
                dtype=torch.float32,
                device=self.device
            ).view(1, -1)  # 确保形状为[1, 72]

            graph_features = self.graph_encoder(sgRNA, dna).to(self.device)
            predicted_activity = self.meta_learner(evolution_tensor, graph_features)
            return predicted_activity.item()

    def save_model(self, path: str, sample_sgRNA: str, sample_dna: str, evolution_distance_dim: int) -> None:
        # 确保动态层已初始化
        self.graph_encoder(sample_sgRNA, sample_dna)  # 初始化动态层

        # 保存模型状态
        model_state = {
            'graph_encoder_state_dict': self.graph_encoder.state_dict(),
            'meta_learner_state_dict': self.meta_learner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'evolution_distance_dim': evolution_distance_dim
        }

        with open(path, 'wb') as f:
            pickle.dump(model_state, f)

        print(f"模型已保存到 {path}")

    def load_model(self, path: str, sample_sgRNA: str, sample_dna: str) -> None:
        """从指定路径加载模型，需要提供样本序列以初始化动态层"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")

        with open(path, 'rb') as f:
            model_state = pickle.load(f)

        # 初始化动态层
        self.graph_encoder(sample_sgRNA, sample_dna)

        # 加载状态字典
        self.graph_encoder.load_state_dict(model_state['graph_encoder_state_dict'])
        self.meta_learner.load_state_dict(model_state['meta_learner_state_dict'])
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])

        print(f"模型已从 {path} 加载")

    def print_model_summary(self) -> None:
        """打印模型架构"""
        print("\n===== 模型架构摘要 =====")
        print("\n1. 图编码器 (SGRNAGraphEncoder):")
        print(self.graph_encoder)

        print("\n2. 元学习器 (MetaLearner):")
        print(self.meta_learner)

        print("\n3. 优化器:")
        print(self.optimizer)
        print("=====================\n")


class SGRNAPredictor:
    """加载预训练模型并进行预测的类"""

    def __init__(self, model_path: str, sample_sgRNA: str, sample_dna: str, evolution_distance_dim: int):
        self.model_path = model_path
        self.sample_sgRNA = sample_sgRNA
        self.sample_dna = sample_dna
        self.evolution_distance_dim = evolution_distance_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型组件
        self.graph_encoder = SGRNAGraphEncoder(output_dim=72)
        self.meta_learner = MetaLearner(evolution_distance_dim=evolution_distance_dim)

        # 加载模型参数
        self._load_model()

        # 设置为评估模式
        self.graph_encoder.to(self.device).eval()
        self.meta_learner.to(self.device).eval()

    def _load_model(self) -> None:
        """从pkl文件加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            model_state = pickle.load(f)

        # 初始化动态层
        self.graph_encoder(self.sample_sgRNA, self.sample_dna)

        # 加载状态字典
        self.graph_encoder.load_state_dict(model_state['graph_encoder_state_dict'])
        self.meta_learner.load_state_dict(model_state['meta_learner_state_dict'])

        print(f"已加载预训练模型: {self.model_path}")

    def predict(self, evolution_distance: list, sgRNA: str, dna: str) -> float:
        """预测sgRNA的打靶活性"""
        with torch.no_grad():
            # 确保进化距离是正确的张量
            evolution_tensor = torch.tensor(
                evolution_distance,
                dtype=torch.float32,
                device=self.device
            ).view(1, -1)  # 确保形状为[1, 72]

            graph_features = self.graph_encoder(sgRNA, dna).to(self.device)
            predicted_activity = self.meta_learner(evolution_tensor, graph_features)
            return predicted_activity.item()


def load_data_from_excel(file_path: str) -> List[Dict[str, Any]]:
    """从Excel文件加载训练数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 验证列名
    required_columns = ['evolution_distance', 'sgRNA', 'dna', 'activity']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Excel文件缺少必要的列: {', '.join(missing_columns)}")

    # 转换数据格式
    data = []
    for _, row in df.iterrows():
        # 解析evolution_distance为列表
        try:
            # 尝试将字符串解析为列表
            if isinstance(row['evolution_distance'], str):
                # 处理类似 "[41.0, 56.0, ...]" 的字符串
                dist_list = eval(row['evolution_distance'])
            else:
                dist_list = row['evolution_distance']

            data.append({
                'evolution_distance': dist_list,
                'sgRNA': str(row['sgRNA']).strip().upper(),
                'dna': str(row['dna']).strip().upper(),
                'activity': float(row['activity'])
            })
        except Exception as e:
            print(f"解析行 {_} 时出错: {e}")
            continue

    print(f"成功从 {file_path} 加载 {len(data)} 条数据")
    return data


def create_example_excel(file_path: str) -> None:
    """创建示例Excel文件，展示数据格式"""
    example_data = {
        'evolution_distance': [
            [41.0, 56.0, 56.0, 60.0, 60.0, 57.0, 57.0, 65.0, 51.0, 54.0, 60.0, 75.0, 66.0, 69.0, 66.0, 72.0, 63.0, 74.0,
             59.0, 74.0, 60.0, 74.0, 67.0, 70.0, 72.0, 74.0, 73.0, 46.0, 41.0, 46.0, 43.0, 63.0, 34.0, 38.0, 0.0, 38.0,
             42.0, 39.0, 43.0, 45.0, 56.0, 55.0, 56.0, 64.0, 56.0, 56.0, 55.0, 57.0, 44.0, 58.0, 65.0, 65.0, 66.0, 67.0,
             59.0, 56.0, 57.0, 43.0, 50.0, 54.0, 56.0, 59.0, 61.0, 60.0, 64.0, 50.0, 47.0, 51.0, 51.0, 61.0, 62.0, 55.0,
             48.0, 48.0, 48.0, 49.0, 58.0, 58.0, 61.0],
            [41.0, 56.0, 56.0, 60.0, 60.0, 57.0, 57.0, 65.0, 51.0, 54.0, 60.0, 75.0, 66.0, 69.0, 66.0, 72.0, 63.0, 74.0,
             59.0, 74.0, 60.0, 74.0, 67.0, 70.0, 72.0, 74.0, 73.0, 46.0, 41.0, 46.0, 43.0, 63.0, 34.0, 38.0, 0.0, 38.0,
             42.0, 39.0, 43.0, 45.0, 56.0, 55.0, 56.0, 64.0, 56.0, 56.0, 55.0, 57.0, 44.0, 58.0, 65.0, 65.0, 66.0, 67.0,
             59.0, 56.0, 57.0, 43.0, 50.0, 54.0, 56.0, 59.0, 61.0, 60.0, 64.0, 50.0, 47.0, 51.0, 51.0, 61.0, 62.0, 55.0,
             48.0, 48.0, 48.0, 49.0, 58.0, 58.0, 61.0]
        ],
        'sgRNA': [
            'GCTAGCTAGCTAGCTAGCTA',
            'ATCGATCGATCGATCGATCG'
        ],
        'dna': [
            'CGATCGATCGATCGATCGAT',
            'TAGCTAGCTAGCTAGCTAGC'
        ],
        'activity': [0.85, 0.72]
    }

    df = pd.DataFrame(example_data)
    df.to_excel(file_path, index=False)
    print(f"已创建示例Excel文件: {file_path}")


# 使用示例
if __name__ == "__main__":
    # # 创建示例Excel文件（如果不存在）
    excel_file = "sgRNA_training_data.xlsx"
    if not os.path.exists(excel_file):
        create_example_excel(excel_file)

    # 尝试从Excel加载数据，失败则使用示例数据
    try:
        training_data = load_data_from_excel(excel_file)
        print(f"使用Excel文件中的{len(training_data)}条数据进行训练")
    except Exception as e:
        print(f"无法从Excel加载数据: {e}，使用示例数据")
        training_data = [
            {
                'evolution_distance': [28.0, 31.0, 31.0, 35.0, 35.0, 32.0, 32.0, 40.0, 0.0, 7.0, 31.0, 46.0, 37.0, 40.0,
                                       37.0, 43.0, 34.0, 45.0, 30.0, 45.0, 31.0, 45.0, 38.0, 41.0, 43.0, 45.0, 44.0,
                                       39.0, 34.0, 39.0, 42.0, 62.0, 35.0, 55.0, 51.0, 53.0, 57.0, 34.0, 38.0, 40.0,
                                       51.0, 50.0, 51.0, 59.0, 51.0, 51.0, 50.0, 52.0, 39.0, 53.0, 60.0, 60.0, 61.0,
                                       62.0, 54.0, 51.0, 52.0, 38.0, 45.0, 49.0, 51.0, 54.0, 56.0, 55.0, 59.0, 45.0,
                                       42.0, 46.0, 46.0, 56.0, 57.0, 50.0, 43.0, 43.0, 43.0, 44.0, 53.0, 53.0, 56.0],
                'sgRNA': 'GCTAGCTAGCTAGCTAGCTA',
                'dna': 'CGATCGATCGATCGATCGAT',
                'activity': 0.85
            },
            {
                'evolution_distance': [28.0, 31.0, 31.0, 35.0, 35.0, 32.0, 32.0, 40.0, 0.0, 7.0, 31.0, 46.0, 37.0, 40.0,
                                       37.0, 43.0, 34.0, 45.0, 30.0, 45.0, 31.0, 45.0, 38.0, 41.0, 43.0, 45.0, 44.0,
                                       39.0, 34.0, 39.0, 42.0, 62.0, 35.0, 55.0, 51.0, 53.0, 57.0, 34.0, 38.0, 40.0,
                                       51.0, 50.0, 51.0, 59.0, 51.0, 51.0, 50.0, 52.0, 39.0, 53.0, 60.0, 60.0, 61.0,
                                       62.0, 54.0, 51.0, 52.0, 38.0, 45.0, 49.0, 51.0, 54.0, 56.0, 55.0, 59.0, 45.0,
                                       42.0, 46.0, 46.0, 56.0, 57.0, 50.0, 43.0, 43.0, 43.0, 44.0, 53.0, 53.0, 56.0],
                'sgRNA': 'ATCGATCGATCGATCGATCG',
                'dna': 'TAGCTAGCTAGCTAGCTAGC',
                'activity': 0.72
            }
        ]

    # 训练模型
    framework = SGRNAMetaLearningFramework(evolution_distance_dim=79)  # 假设进化距离维度为79

    # 训练几个轮次
    for epoch in range(5):
        loss = framework.train_step(training_data)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    # 打印模型架构
    framework.print_model_summary()

    # 保存模型
    model_path = "meta-crispr.pkl"
    sample_sgRNA = training_data[0]['sgRNA']
    sample_dna = training_data[0]['dna']
    framework.save_model(model_path, sample_sgRNA, sample_dna, evolution_distance_dim=79)

    # 使用预测类进行预测
    predictor = SGRNAPredictor(
        model_path,
        sample_sgRNA=sample_sgRNA,
        sample_dna=sample_dna,
        evolution_distance_dim=79
    )

    # 预测
    prediction = predictor.predict(
        evolution_distance=training_data[0]['evolution_distance'],
        sgRNA=sample_sgRNA,
        dna=sample_dna
    )

    print(f"使用预测类预测的活性: {prediction:.4f}")