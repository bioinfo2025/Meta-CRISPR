from ete3 import Tree
import numpy as np
from sklearn.manifold import MDS


def generate_evolutionary_embeddings(newick_file, n_components=32):
    # 加载Newick树
    tree = Tree(newick_file)

    # 获取所有叶节点（物种）
    species_list = [leaf.name for leaf in tree.get_leaves()]

    # 构建距离矩阵
    n_species = len(species_list)
    distance_matrix = np.zeros((n_species, n_species))

    for i in range(n_species):
        for j in range(n_species):
            node_i = tree & species_list[i]
            node_j = tree & species_list[j]
            distance_matrix[i, j] = node_i.get_distance(node_j)

    # 使用
    # 降维
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    embeddings = mds.fit_transform(distance_matrix)

    # 创建物种到嵌入的字典
    species_to_embedding = {sp: embeddings[i] for i, sp in enumerate(species_list)}

    return species_to_embedding