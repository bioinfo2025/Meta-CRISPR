import numpy as np
import pandas as pd
from ete3 import Tree, TreeStyle, NodeStyle, TextFace
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform


# ========================
# Part 1: 加载并验证进化树
# ========================
def load_phylogenetic_tree(nwk_path):
    """加载Newick树文件并验证拓扑结构"""
    try:
        t = Tree(nwk_path)
        print(f"Loaded tree with {len(t)} species. Rooted: {t.is_rooted}")

        # 检查与NCBI Taxonomy的一致性（示例物种）
        expected_species = ['Homo_sapiens', 'Mus_musculus', 'Danio_rerio']
        for leaf in t.iter_leaves():
            if leaf.name not in expected_species:
                print(f"Warning: {leaf.name} not in reference taxonomy")
        return t
    except Exception as e:
        raise ValueError(f"Error loading Newick file: {str(e)}")


# ============================
# Part 2: 计算进化距离矩阵
# ============================
def compute_distance_matrix(tree):
    """计算物种间进化距离矩阵"""
    leaves = tree.get_leaf_names()
    n = len(leaves)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            node_i = tree & leaves[i]
            node_j = tree & leaves[j]
            dist_matrix[i, j] = node_i.get_distance(node_j)

    # 转换为对称矩阵并验证
    if not np.allclose(dist_matrix, dist_matrix.T):
        raise ValueError("Distance matrix is not symmetric")
    return pd.DataFrame(dist_matrix, index=leaves, columns=leaves)


# ============================
# Part 3: 生成低维进化嵌入
# ============================
def generate_evolutionary_embeddings(dist_matrix, n_components=32):
    """使用MDS生成进化嵌入向量"""
    mds = MDS(n_components=n_components,
              dissimilarity='precomputed',
              random_state=42,
              normalized_stress='auto')

    embeddings = mds.fit_transform(dist_matrix)
    stress = mds.stress_
    print(f"MDS completed with stress: {stress:.4f}")

    return pd.DataFrame(embeddings,
                        index=dist_matrix.index,
                        columns=[f"PC{i + 1}" for i in range(n_components)])


# ============================
# Part 4: 可视化模块
# ============================
def visualize_phylogenetic_tree(tree, output_path="tree.pdf"):
    """生成出版级进化树可视化"""
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.branch_vertical_margin = 10
    ts.scale = 100  # 每单位分支长度对应像素

    # 设置节点样式
    for n in tree.traverse():
        ns = NodeStyle()
        ns["size"] = 0  # 隐藏节点标记
        if n.is_leaf():
            n.add_face(TextFace(n.name, fsize=8), column=0)
        n.set_style(ns)

    tree.render(output_path, tree_style=ts)
    print(f"Tree visualization saved to {output_path}")


def plot_embedding_space(embeddings):
    """可视化MDS嵌入空间"""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings['PC1'], y=embeddings['PC2'],
                    hue=embeddings.index.str.split('_').str[0],  # 按属着色
                    palette='viridis',
                    s=100,
                    edgecolor='w')

    plt.xlabel(f"PC1 ({mds.explained_variance_ratio_[0]:.1%} Variance)")
    plt.ylabel(f"PC2 ({mds.explained_variance_ratio_[1]:.1%} Variance)")
    plt.title("Evolutionary Embedding Space")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("evolutionary_embedding.pdf", dpi=300)
    plt.close()


# ============================
# 主流程
# ============================
if __name__ == "__main__":
    # 输入参数
    NWK_PATH = "species_tree.nwk"
    EMBEDDING_DIM = 32

    # 1. 加载并验证树
    phylo_tree = load_phylogenetic_tree(NWK_PATH)

    # 2. 计算距离矩阵
    distance_df = compute_distance_matrix(phylo_tree)
    distance_df.to_csv("evolutionary_distances.csv")

    # 3. 生成嵌入向量
    embedding_df = generate_evolutionary_embeddings(distance_df, EMBEDDING_DIM)
    embedding_df.to_csv("species_embeddings.csv")

    # 4. 可视化
    visualize_phylogenetic_tree(phylo_tree)
    plot_embedding_space(embedding_df)