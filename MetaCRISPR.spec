import os
from PyInstaller.utils.hooks import collect_all

datas = []
extra_binaries = []

# 收集 torch_geometric 依赖
geometric_data, geometric_binaries, geometric_hiddenimports = collect_all('torch_geometric')
datas += geometric_data
extra_binaries += geometric_binaries

# 添加 PyQt5 插件（如果需要）
#datas += [('/opt/anaconda3/envs/meta-crispr/lib/python3.9/site-packages/PyQt5/Qt5/plugins/platforms', 'platforms')]

datas += [('species_encoding.json', '.')]
datas += [('meta-crispr.pkl', '.')]  # 将文件复制到应用根目录

# 从环境变量中获取conda环境路径（若未激活Conda环境，默认为空字符串）
conda_prefix = os.environ.get('CONDA_PREFIX', '')  # 关键行：定义conda_prefix变量
# 配置环境变量
env_vars = {
    # Qt插件路径（解决Qt库版本冲突）
    "QT_QPA_PLATFORM_PLUGIN_PATH": os.path.join(conda_prefix, "plugins/platforms"),
    # macOS动态库路径（可选，根据需要保留）
    "DYLD_FRAMEWORK_PATH": os.path.join(conda_prefix, "lib") if conda_prefix else "",
    # 禁用PyTorch JIT编译（避免警告）
    "PYTORCH_JIT": "0"
}


a = Analysis(
    ['MetaCrisprGUI.py'],
    pathex=['/Users/anna/PycharmProjects/Meta-CRISPR2'],  # 替换为实际路径
    binaries=extra_binaries,
    datas=datas,
    hiddenimports=geometric_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
    # 注入环境变量到Analysis对象
    env=env_vars,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MetaCRISPR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 调试完成后改为 False
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 🌟 添加 BUNDLE 部分（生成 macOS .app 包的关键）
app = BUNDLE(
    exe,  # 引用前面定义的 EXE 对象
    name='MetaCRISPR.app',
    icon=None,  # 可选：指定应用图标 (.icns 文件)
    bundle_identifier='com.yourcompany.metacrispr',  # 可选：应用标识符
    info_plist={
        'NSHighResolutionCapable': 'True',  # 支持高分辨率显示
        'CFBundleShortVersionString': '1.0.0',  # 版本号
        'LSUIElement': '0',  # 0 表示显示在 Dock 中，1 表示不显示
    }
)