import matplotlib.pyplot as plt
from ete3 import Tree, TreeStyle, NodeStyle, faces, AttrFace, CircleFace
import re


def validate_nwk_file(nwk_file):
    """检查并简单修复NWK文件格式问题"""
    with open(nwk_file, 'r') as f:
        content = f.read().strip()

    # 检查括号平衡
    stack = []
    for i, char in enumerate(content):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if not stack:
                print(f"错误: 未匹配的右括号在位置 {i}")
                # 尝试简单修复：添加左括号
                content = '(' + content
                stack.append(0)
            else:
                stack.pop()

    # 添加缺失的右括号
    while stack:
        print(f"错误: 未匹配的左括号在位置 {stack.pop()}")
        content += ')'

    # 确保以分号结尾
    if not content.endswith(';'):
        content += ';'

    # 保存修复后的文件
    fixed_nwk_file = nwk_file.replace('.nwk', '_fixed.nwk')
    with open(fixed_nwk_file, 'w') as f:
        f.write(content)

    print(f"已修复并保存至: {fixed_nwk_file}")
    return fixed_nwk_file


def generate_nature_style_circular_tree(nwk_file, output_path="circular_tree.png", dpi=300):
    """生成符合《Nature》风格的环形进化树"""
    try:
        # 尝试直接加载
        tree = Tree(nwk_file, format=1)
    except Exception as e:
        print(f"加载失败: {e}")
        print("尝试修复NWK格式...")
        fixed_nwk_file = validate_nwk_file(nwk_file)
        # 尝试使用不同格式加载修复后的文件
        try:
            tree = Tree(fixed_nwk_file, format=8)  # 格式8忽略分支长度
            print("使用格式8成功加载修复后的文件")
        except Exception as e2:
            print(f"仍然无法加载: {e2}")
            print("请手动检查NWK文件格式")
            return

    # 后续代码保持不变...
    node_style = NodeStyle()
    node_style["size"] = 0
    node_style["vt_line_width"] = 1
    node_style["hz_line_width"] = 1
    node_style["line_color"] = "#333333"

    def set_leaf_style(node):
        if node.is_leaf():
            face = AttrFace("name", fgcolor="#000000", fsize=8, bold=True, fontfamily="Arial")
            faces.add_face_to_node(face, node, 0, position="branch-right")

    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.mode = "c"
    ts.arc_start = 180
    ts.arc_span = 360
    ts.margin_top = 20
    ts.margin_bottom = 20
    ts.margin_left = 20
    ts.margin_right = 20
    ts.background_color = "#FFFFFF"
    ts.draw_guiding_lines = False
    ts.branch_vertical_margin = 4

    for node in tree.traverse():
        node.set_style(node_style)
        if node.is_leaf():
            set_leaf_style(node)

    tree.set_outgroup(tree.get_leaves()[0])

    tree.render(output_path, tree_style=ts, dpi=dpi, w=800, h=800)
    print(f"已生成《Nature》风格环形树：{output_path}")


# 示例用法
if __name__ == "__main__":
    nwk_file = "valid_tree_fixed.nwk"  # 替换为你的NWK文件路径
    generate_nature_style_circular_tree(nwk_file)