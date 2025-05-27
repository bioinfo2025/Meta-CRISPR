import wx
import wx.adv
import wx.lib.scrolledpanel as scrolled
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import numpy as np
import random
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class MainFrame(wx.Frame):
    """Meta-CRISPR工具的主窗口"""

    def __init__(self, parent=None, title="Meta-CRISPR - sgRNA打靶活性分析工具"):
        super(MainFrame, self).__init__(parent, title=title, size=(1200, 800))

        # 设置图标
        self.SetIcon(wx.Icon(wx.ArtProvider.GetBitmap(wx.ART_INFORMATION, wx.ART_OTHER, (16, 16))))

        # 创建菜单栏
        self.create_menu()

        # 创建状态栏
        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("就绪")

        # 创建主面板
        self.main_panel = wx.Panel(self)

        # 创建主Sizer，使用FlexGridSizer便于布局
        self.main_sizer = wx.FlexGridSizer(rows=1, cols=2, vgap=10, hgap=10)
        self.main_sizer.AddGrowableCol(0, 1)
        self.main_sizer.AddGrowableCol(1, 2)
        self.main_sizer.AddGrowableRow(0, 1)

        # 创建输入面板
        self.create_input_panel()

        # 创建结果面板
        self.create_result_panel()

        # 添加面板到主Sizer
        self.main_sizer.Add(self.input_panel, 1, wx.EXPAND | wx.ALL, 5)
        self.main_sizer.Add(self.result_panel, 1, wx.EXPAND | wx.ALL, 5)

        # 设置主Sizer
        self.main_panel.SetSizer(self.main_sizer)

        # 布局和调整大小
        self.main_sizer.Fit(self)
        self.Center()
        self.Show()

    def create_menu(self):
        """创建菜单栏"""
        menubar = wx.MenuBar()

        # 文件菜单
        file_menu = wx.Menu()
        new_item = file_menu.Append(wx.ID_NEW, "新建\tCtrl+N", "创建新的分析")
        open_item = file_menu.Append(wx.ID_OPEN, "打开\tCtrl+O", "打开保存的分析")
        file_menu.AppendSeparator()
        save_item = file_menu.Append(wx.ID_SAVE, "保存\tCtrl+S", "保存当前分析")
        save_as_item = file_menu.Append(wx.ID_SAVEAS, "另存为\tCtrl+Shift+S", "另存为")
        file_menu.AppendSeparator()
        exit_item = file_menu.Append(wx.ID_EXIT, "退出\tCtrl+Q", "退出程序")

        # 编辑菜单
        edit_menu = wx.Menu()
        copy_item = edit_menu.Append(wx.ID_COPY, "复制\tCtrl+C", "复制选中内容")
        paste_item = edit_menu.Append(wx.ID_PASTE, "粘贴\tCtrl+V", "粘贴内容")
        edit_menu.AppendSeparator()
        preferences_item = edit_menu.Append(wx.ID_PREFERENCES, "偏好设置", "设置程序偏好")

        # 分析菜单
        analysis_menu = wx.Menu()
        run_analysis_item = analysis_menu.Append(wx.ID_ANY, "运行分析\tF5", "执行sgRNA打靶活性分析")
        analysis_menu.AppendSeparator()
        batch_mode_item = analysis_menu.Append(wx.ID_ANY, "批量模式", "批量处理多个sgRNA")

        # 帮助菜单
        help_menu = wx.Menu()
        about_item = help_menu.Append(wx.ID_ABOUT, "关于", "关于Meta-CRISPR工具")
        help_item = help_menu.Append(wx.ID_HELP, "帮助\tF1", "显示帮助文档")

        # 添加菜单到菜单栏
        menubar.Append(file_menu, "文件")
        menubar.Append(edit_menu, "编辑")
        menubar.Append(analysis_menu, "分析")
        menubar.Append(help_menu, "帮助")

        # 设置菜单栏
        self.SetMenuBar(menubar)

        # 绑定菜单事件
        self.Bind(wx.EVT_MENU, self.on_new, new_item)
        self.Bind(wx.EVT_MENU, self.on_open, open_item)
        self.Bind(wx.EVT_MENU, self.on_save, save_item)
        self.Bind(wx.EVT_MENU, self.on_save_as, save_as_item)
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)
        self.Bind(wx.EVT_MENU, self.on_copy, copy_item)
        self.Bind(wx.EVT_MENU, self.on_paste, paste_item)
        self.Bind(wx.EVT_MENU, self.on_preferences, preferences_item)
        self.Bind(wx.EVT_MENU, self.on_run_analysis, run_analysis_item)
        self.Bind(wx.EVT_MENU, self.on_batch_mode, batch_mode_item)
        self.Bind(wx.EVT_MENU, self.on_about, about_item)
        self.Bind(wx.EVT_MENU, self.on_help, help_item)

    def create_input_panel(self):
        """创建输入面板"""
        self.input_panel = wx.Panel(self.main_panel, style=wx.BORDER_SUNKEN)
        self.input_panel.SetBackgroundColour(wx.Colour(245, 245, 245))

        # 创建垂直Sizer
        input_sizer = wx.BoxSizer(wx.VERTICAL)

        # 创建标题
        title_label = wx.StaticText(self.input_panel, label="输入参数", style=wx.ALIGN_CENTER)
        title_font = wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title_label.SetFont(title_font)
        input_sizer.Add(title_label, 0, wx.ALL | wx.EXPAND, 10)
        input_sizer.Add(wx.StaticLine(self.input_panel), 0, wx.EXPAND | wx.ALL, 5)

        # 创建输入字段
        field_sizer = wx.FlexGridSizer(rows=5, cols=2, vgap=10, hgap=10)

        # sgRNA序列
        field_sizer.Add(wx.StaticText(self.input_panel, label="sgRNA序列:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        self.sgrna_text = wx.TextCtrl(self.input_panel, style=wx.TE_MULTILINE)
        self.sgrna_text.SetToolTip("输入20个碱基的sgRNA序列（例如：GCTAGCTAGCTAGCTAGCTA）")
        field_sizer.Add(self.sgrna_text, 1, wx.EXPAND | wx.RIGHT, 5)

        # 目标序列
        field_sizer.Add(wx.StaticText(self.input_panel, label="目标序列:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        self.target_text = wx.TextCtrl(self.input_panel, style=wx.TE_MULTILINE)
        self.target_text.SetToolTip("输入目标DNA序列（例如：GCTAGCTAGCTAGCTAGCTAGG）")
        field_sizer.Add(self.target_text, 1, wx.EXPAND | wx.RIGHT, 5)

        # PAM序列
        field_sizer.Add(wx.StaticText(self.input_panel, label="PAM序列:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        pam_choices = ["NGG", "NG", "NAG", "NNGRRT", "NNAGAAW"]
        self.pam_combo = wx.ComboBox(self.input_panel, choices=pam_choices, style=wx.CB_DROPDOWN)
        self.pam_combo.SetValue("NGG")
        self.pam_combo.SetToolTip("选择或输入PAM序列（例如：NGG）")
        field_sizer.Add(self.pam_combo, 1, wx.EXPAND | wx.RIGHT, 5)

        # 物种
        field_sizer.Add(wx.StaticText(self.input_panel, label="物种:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        species_choices = ["人类(Homo sapiens)", "小鼠(Mus musculus)", "大鼠(Rattus norvegicus)", "斑马鱼(Danio rerio)",
                           "果蝇(Drosophila melanogaster)", "拟南芥(Arabidopsis thaliana)", "其他"]
        self.species_combo = wx.ComboBox(self.input_panel, choices=species_choices, style=wx.CB_DROPDOWN)
        self.species_combo.SetValue("人类(Homo sapiens)")
        self.species_combo.SetToolTip("选择目标物种")
        field_sizer.Add(self.species_combo, 1, wx.EXPAND | wx.RIGHT, 5)

        # 分析参数
        field_sizer.Add(wx.StaticText(self.input_panel, label="分析参数:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        param_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # 脱靶预测算法
        algorithm_choices = ["CRISPRoff", "CCTop", "Cas-OFFinder", "MIT算法"]
        self.algorithm_combo = wx.ComboBox(self.input_panel, choices=algorithm_choices, style=wx.CB_DROPDOWN)
        self.algorithm_combo.SetValue("CRISPRoff")
        self.algorithm_combo.SetToolTip("选择脱靶预测算法")
        param_sizer.Add(self.algorithm_combo, 1, wx.RIGHT, 5)

        # 错配数
        mismatch_label = wx.StaticText(self.input_panel, label="最大错配数:")
        param_sizer.Add(mismatch_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 5)
        self.mismatch_spin = wx.SpinCtrl(self.input_panel, min=0, max=10, initial=3)
        self.mismatch_spin.SetToolTip("设置允许的最大错配数")
        param_sizer.Add(self.mismatch_spin, 0, wx.RIGHT, 5)

        field_sizer.Add(param_sizer, 1, wx.EXPAND | wx.RIGHT, 5)

        # 设置字段Sizer的列比例
        field_sizer.AddGrowableCol(1, 1)

        # 添加字段Sizer到主Sizer
        input_sizer.Add(field_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # 添加按钮Sizer
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # 运行按钮
        self.run_button = wx.Button(self.input_panel, label="运行分析", size=(120, 40))
        self.run_button.SetToolTip("执行sgRNA打靶活性分析和脱靶预测")
        self.run_button.Bind(wx.EVT_BUTTON, self.on_run_analysis)
        button_sizer.Add(self.run_button, 0, wx.ALL, 5)

        # 重置按钮
        self.reset_button = wx.Button(self.input_panel, label="重置", size=(120, 40))
        self.reset_button.SetToolTip("清空所有输入字段")
        self.reset_button.Bind(wx.EVT_BUTTON, self.on_reset)
        button_sizer.Add(self.reset_button, 0, wx.ALL, 5)

        # 添加按钮Sizer到主Sizer
        input_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, 15)

        # 设置输入面板的Sizer
        self.input_panel.SetSizer(input_sizer)

    def create_result_panel(self):
        """创建结果面板"""
        self.result_panel = scrolled.ScrolledPanel(self.main_panel, style=wx.BORDER_SUNKEN)
        self.result_panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.result_panel.SetupScrolling()

        # 创建垂直Sizer
        result_sizer = wx.BoxSizer(wx.VERTICAL)

        # 创建标题
        title_label = wx.StaticText(self.result_panel, label="分析结果", style=wx.ALIGN_CENTER)
        title_font = wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title_label.SetFont(title_font)
        result_sizer.Add(title_label, 0, wx.ALL | wx.EXPAND, 10)
        result_sizer.Add(wx.StaticLine(self.result_panel), 0, wx.EXPAND | wx.ALL, 5)

        # 创建笔记本控件用于标签页
        self.notebook = wx.Notebook(self.result_panel)

        # 创建打靶活性分析标签页
        self.activity_panel = wx.Panel(self.notebook)
        activity_sizer = wx.BoxSizer(wx.VERTICAL)

        # 创建活性评分区域
        score_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # 活性评分标题
        score_title = wx.StaticText(self.activity_panel, label="打靶活性评分:")
        score_font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        score_title.SetFont(score_font)
        score_sizer.Add(score_title, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)

        # 活性评分值
        self.score_value = wx.StaticText(self.activity_panel, label="--")
        score_value_font = wx.Font(24, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.score_value.SetFont(score_value_font)
        score_sizer.Add(self.score_value, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)

        # 活性评分描述
        self.score_desc = wx.StaticText(self.activity_panel, label="")
        self.score_desc.SetForegroundColour(wx.Colour(100, 100, 100))
        score_sizer.Add(self.score_desc, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 20)

        activity_sizer.Add(score_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 添加评分条
        self.score_gauge = wx.Gauge(self.activity_panel, range=100, size=(-1, 30))
        activity_sizer.Add(self.score_gauge, 0, wx.EXPAND | wx.ALL, 10)

        # 添加序列比对结果
        alignment_title = wx.StaticText(self.activity_panel, label="序列比对:")
        alignment_title.SetFont(score_font)
        activity_sizer.Add(alignment_title, 0, wx.ALL, 5)

        self.alignment_text = wx.TextCtrl(self.activity_panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        activity_sizer.Add(self.alignment_text, 1, wx.EXPAND | wx.ALL, 5)

        # 添加特征重要性图
        importance_title = wx.StaticText(self.activity_panel, label="特征重要性分析:")
        importance_title.SetFont(score_font)
        activity_sizer.Add(importance_title, 0, wx.ALL, 5)

        # 创建Matplotlib图形
        self.importance_fig = plt.figure(figsize=(8, 4))
        self.importance_canvas = FigureCanvas(self.activity_panel, -1, self.importance_fig)
        activity_sizer.Add(self.importance_canvas, 1, wx.EXPAND | wx.ALL, 5)

        # 设置活性面板的Sizer
        self.activity_panel.SetSizer(activity_sizer)

        # 创建脱靶图谱标签页
        self.offtarget_panel = wx.Panel(self.notebook)
        offtarget_sizer = wx.BoxSizer(wx.VERTICAL)

        # 添加脱靶位点表格
        offtarget_title = wx.StaticText(self.offtarget_panel, label="预测脱靶位点:")
        offtarget_title.SetFont(score_font)
        offtarget_sizer.Add(offtarget_title, 0, wx.ALL, 5)

        # 创建脱靶位点列表控件
        self.offtarget_list = wx.ListCtrl(self.offtarget_panel, style=wx.LC_REPORT | wx.LC_HRULES | wx.LC_VRULES)
        self.offtarget_list.InsertColumn(0, "排名", width=80)
        self.offtarget_list.InsertColumn(1, "序列", width=250)
        self.offtarget_list.InsertColumn(2, "位置", width=120)
        self.offtarget_list.InsertColumn(3, "链", width=80)
        self.offtarget_list.InsertColumn(4, "相似度", width=100)
        self.offtarget_list.InsertColumn(5, "活性评分", width=100)
        offtarget_sizer.Add(self.offtarget_list, 1, wx.EXPAND | wx.ALL, 5)

        # 添加脱靶分布热图
        heatmap_title = wx.StaticText(self.offtarget_panel, label="全基因组脱靶分布:")
        heatmap_title.SetFont(score_font)
        offtarget_sizer.Add(heatmap_title, 0, wx.ALL, 5)

        # 创建Matplotlib图形
        self.heatmap_fig = plt.figure(figsize=(10, 6))
        self.heatmap_canvas = FigureCanvas(self.offtarget_panel, -1, self.heatmap_fig)
        offtarget_sizer.Add(self.heatmap_canvas, 1, wx.EXPAND | wx.ALL, 5)

        # 设置脱靶面板的Sizer
        self.offtarget_panel.SetSizer(offtarget_sizer)

        # 创建报告标签页
        self.report_panel = wx.Panel(self.notebook)
        report_sizer = wx.BoxSizer(wx.VERTICAL)

        # 添加报告文本区域
        report_title = wx.StaticText(self.report_panel, label="分析报告:")
        report_title.SetFont(score_font)
        report_sizer.Add(report_title, 0, wx.ALL, 5)

        self.report_text = wx.TextCtrl(self.report_panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        report_sizer.Add(self.report_text, 1, wx.EXPAND | wx.ALL, 5)

        # 添加导出按钮
        export_button = wx.Button(self.report_panel, label="导出报告")
        export_button.Bind(wx.EVT_BUTTON, self.on_export_report)
        report_sizer.Add(export_button, 0, wx.ALIGN_RIGHT | wx.ALL, 10)

        # 设置报告面板的Sizer
        self.report_panel.SetSizer(report_sizer)