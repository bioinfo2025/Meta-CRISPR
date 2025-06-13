import json
import os
import sys

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 在导入pyplot之前设置

import matplotlib.pyplot as plt
import seaborn as sns


class PhylogeneticEncoder:
    """基于进化树的物种编码器，支持多种编码方式和直接通过物种名称查询"""

    def __init__(self, json_path):
        """初始化并加载进化树编码数据"""
        self.json_path = json_path
        self.data = self._load_data()
        self.species_list = list(self.data.keys())
        self._build_hierarchy_map()
        self._build_distance_matrix()
        self._create_species_index()

    def _load_data(self):
        """加载JSON格式的进化树数据"""
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def _create_species_index(self):
        """创建物种名称到矩阵索引的映射"""
        self.species_to_index = {name: idx for idx, name in enumerate(self.species_list)}

    def _build_hierarchy_map(self):
        """构建层次结构映射"""
        self.node_levels = {}
        for species, info in self.data.items():
            hierarchy = info['hierarchy']
            for level, node in enumerate(hierarchy):
                if node not in self.node_levels:
                    self.node_levels[node] = level

    def _build_distance_matrix(self):
        """构建物种间的进化距离矩阵"""
        n = len(self.species_list)
        self.distance_matrix = np.zeros((n, n))

        for i, sp1 in enumerate(self.species_list):
            for j, sp2 in enumerate(self.species_list):
                if i == j:
                    continue
                self.distance_matrix[i, j] = self._calculate_distance(sp1, sp2)

    def _calculate_distance(self, species1, species2):
        """计算两个物种之间的进化距离"""
        path1 = self.data[species1]['hierarchy']
        path2 = self.data[species2]['hierarchy']

        # 找到最近共同祖先 (LCA)
        lca = None
        for node in reversed(path1):
            if node in path2:
                lca = node
                break

        if lca is None:
            return float('inf')

        # 计算从LCA到两个物种的总距离
        lca_index1 = path1.index(lca)
        lca_index2 = path2.index(lca)

        # 累加分支长度
        distance = sum(self.data[species1]['branch_length'] for _ in range(lca_index1, len(path1) - 1))
        distance += sum(self.data[species2]['branch_length'] for _ in range(lca_index2, len(path2) - 1))

        return distance

    def get_distance_encoding(self, species_name):
        """根据物种名称获取进化距离编码"""
        if species_name not in self.species_to_index:
            raise ValueError(f"物种名称 '{species_name}' 不存在于数据中")

        index = self.species_to_index[species_name]
        return self.distance_matrix[index, :].tolist()

    def get_pairwise_distance(self, species1, species2):
        """获取两个物种之间的进化距离"""
        if species1 not in self.species_to_index or species2 not in self.species_to_index:
            missing = [s for s in [species1, species2] if s not in self.species_to_index]
            raise ValueError(f"物种名称 '{missing[0]}' 不存在于数据中")

        i = self.species_to_index[species1]
        j = self.species_to_index[species2]
        return self.distance_matrix[i, j]

    def encode_distance_vector(self):
        """基于进化距离的向量编码"""
        encoded_data = {}
        for i, species in enumerate(self.species_list):
            encoded_data[species] = self.distance_matrix[i, :].tolist()
        return encoded_data

    def encode_hierarchy_vector(self, max_levels=None):
        """基于层次结构的向量编码"""
        if max_levels is None:
            max_levels = max(self.node_levels.values()) + 1

        encoded_data = {}
        for species, info in self.data.items():
            hierarchy = info['hierarchy']
            vector = []

            # 为每个层级创建特征
            for level in range(max_levels):
                if level < len(hierarchy):
                    vector.append(hierarchy[level])
                else:
                    vector.append("None")  # 填充缺失层级

            encoded_data[species] = vector

        # 转换为独热编码
        return self._one_hot_encode(encoded_data)

    def _one_hot_encode(self, encoded_data):
        """对分类特征进行独热编码"""
        # 将数据转换为适合sklearn的格式
        species_names = list(encoded_data.keys())
        feature_matrix = np.array([encoded_data[sp] for sp in species_names])

        # 应用独热编码
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoded_matrix = encoder.fit_transform(feature_matrix).toarray()

        # 转换回字典格式
        encoded_dict = {}
        for i, species in enumerate(species_names):
            encoded_dict[species] = encoded_matrix[i].tolist()

        return encoded_dict

    def encode_branch_lengths(self):
        """基于分支长度的编码"""
        encoded_data = {}
        for species, info in self.data.items():
            # 使用从根到该物种的所有分支长度作为特征
            branch_lengths = [info['branch_length']]

            # 这里简化处理，实际中可能需要从hierarchy中提取完整的分支长度序列
            encoded_data[species] = branch_lengths

        return encoded_data


    def visualize_distance_matrix(self, output_file=None, show_plot=False, figsize=(12, 10),
                                  cmap=None, annot_size=8, dendrogram=True,
                                  title="Species Evolutionary Distance Matrix", dpi=300,
                                  linkage_method='ward', distance_threshold=None):
        """
        可视化物种间的进化距离矩阵，支持高度自定义的热图和聚类参数

        Parameters:
        - linkage_method: 聚类方法（默认ward）
        - distance_threshold: 树状图切割阈值（None表示不切割）
        """
        # 构建DataFrame
        df = pd.DataFrame(
            self.distance_matrix,
            index=self.species_list,
            columns=self.species_list
        )

        # --------------------------- 自定义颜色映射 ---------------------------
        custom_colors = [

            "#f7acad", "#edd7d7", "#c79cc8", "#98d6d5", "#b66fb3", '#7191c6'
        ]
        # 创建连续颜色映射（可根据需求调整为分段映射）
        cmap = ListedColormap(sns.color_palette(custom_colors).as_hex())
            #if cmap is None else cmap

        # --------------------------- 绘图设置 ---------------------------
        plt.figure(figsize=figsize, dpi=dpi)
        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 16,
            "axes.labelsize": 12
        })

        if dendrogram:
            # --------------------------- 聚类热图 ---------------------------
            g = sns.clustermap(
                df,
                annot=True,
                fmt='.2f',
                annot_kws={"size": annot_size, "color": "darkslategray"},  # 标注字体颜色
                cmap=cmap,
                linewidths=0.3,
                figsize=figsize,
                cbar_kws={
                    "label": "Evolutionary Distance",
                    "shrink": 0.85,  # 颜色条缩放比例
                    "ticks": np.linspace(df.min().min(), df.max().max(), 5)  # 自定义颜色条刻度
                },
                method=linkage_method,  # 聚类方法
                metric='euclidean',  # 距离度量
                dendrogram_ratio=0.15,  # 树状图宽度比例
                yticklabels=True,
                xticklabels=True
            )

            # --------------------------- 树状图优化 ---------------------------
            if distance_threshold:
                g.ax_heatmap.collections[0].set_clim(vmin=0, vmax=distance_threshold)
                g.ax_col_dendrogram.hlines(distance_threshold, 0, g.dendrogram_col.reordered_ind.size, colors='r',
                                           linestyles='--')
                g.ax_row_dendrogram.vlines(distance_threshold, 0, g.dendrogram_row.reordered_ind.size, colors='r',
                                           linestyles='--')

            # --------------------------- 标题与布局 ---------------------------
            g.fig.suptitle(title, y=1.02, fontweight='bold')  # 加粗标题
            g.ax_heatmap.set_xlabel('Species', labelpad=15)
            g.ax_heatmap.set_ylabel('Species', labelpad=15)
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        else:
            # --------------------------- 传统热图 ---------------------------
            ax = sns.heatmap(
                df,
                annot=True,
                fmt='.2f',
                annot_kws={"size": annot_size},
                cmap=cmap,
                linewidths=0.3,
                cbar_kws={
                    "label": "evolutionary distance",
                    "shrink": 0.9,
                    "orientation": "vertical"
                }
            )
            ax.set_title(title, pad=20)
            ax.set_xlabel('Species', labelpad=10)
            ax.set_ylabel('Species', labelpad=10)
            plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

        # --------------------------- 输出与显示 ---------------------------
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0.5)
            plt.close()
        elif show_plot:
            plt.tight_layout()
            plt.show()

    def save_encoding_to_json(self, encoded_data, output_path):
        """将编码结果保存到JSON文件"""
        with open(output_path, 'w') as f:
            json.dump(encoded_data, f, indent=2)

    def combine_encodings(self, encoding_types=['distance', 'hierarchy']):
        """组合多种编码方式"""
        combined = {}

        for species in self.species_list:
            features = []

            if 'distance' in encoding_types:
                features += self.get_distance_encoding(species)

            if 'hierarchy' in encoding_types:
                hierarchy_dict = self.encode_hierarchy_vector()
                features += hierarchy_dict[species]

            if 'branch' in encoding_types:
                branch_dict = self.encode_branch_lengths()
                features += branch_dict[species]

            combined[species] = features

        return combined


def get_resource_path(relative_path):
    """获取资源文件的绝对路径，支持打包后的应用和直接运行的脚本"""
    try:
        # PyInstaller创建临时文件夹并设置sys._MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # 直接运行脚本时的情况
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


if __name__ == '__main__':
    # 初始化编码器
    encoder = PhylogeneticEncoder(get_resource_path("species_encoding.json"))



    #4. 可视化进化距离矩阵
    encoder.visualize_distance_matrix(
        output_file="distance_matrix.png",
        cmap="YlGnBu",
        title="Species Evolutionary Distance Matrix"
    )


    # 1. 获取单个物种的进化距离编码
    human_encoding = encoder.get_distance_encoding("Homo_sapiens")
    print(f"人类的进化距离编码长度: {len(human_encoding)}")

    # 2. 获取两个物种之间的进化距离
    #distance = encoder.get_pairwise_distance("Homo_sapiens", "")
    #print(f"人类与黑猩猩的进化距离: {distance:.4f}")

    # 3. 获取所有物种的层次结构编码
    #hierarchy_encodings = encoder.encode_hierarchy_vector()
    #print(f"层次结构编码维度: {len(hierarchy_encodings['Homo_sapiens'])}")

    # 5. 组合多种编码方式
    #combined = encoder.combine_encodings(['distance', 'hierarchy'])
    #print(f"组合编码维度: {len(combined['Homo_sapiens'])}")

    # 6. 保存编码结果到JSON
    #encoder.save_encoding_to_json(combined, "combined_encoding.json")