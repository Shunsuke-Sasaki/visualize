import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

# --- CSVファイルのパス定義 ---
csv_files = {
    'LR': '/Users/sshunsuke/Downloads/rmse_statistics_linear.csv',
    'NN': '/Users/sshunsuke/Downloads/rmse_statistics_nn.csv',
    'SR': '/Users/sshunsuke/Downloads/rmse_sr.csv',
}

# --- RMSEの単位（正規化後は単位情報は表示しません） ---
rmse_units = {
    'dGs': 'kcal/mol',
    'Ebd': 'MV/m',
    'log10(lifetime)': 'log(year)',
    'logD': 'logD',
    'logP': 'logP',
    'logS': 'logS',
    'pKaA': 'pKa',
    'pKaB': 'pKa',
    'RI': 'RI',
    'Tb': 'K',
    'Tm': 'K',
}

# --- 対象とする予測対象（target）のリスト ---
targets = list(rmse_units.keys())

# --- 保存先フォルダの指定 ---
save_dir = "/Users/sshunsuke/Downloads/"
os.makedirs(save_dir, exist_ok=True)

# --- 各CSVファイルの読み込み ---
dfs = {}
for model, file_path in csv_files.items():
    dfs[model] = pd.read_csv(file_path)

# --- 各モデル・RMSEタイプごとの正規化済み値およびエラーデータを格納する辞書の初期化 ---
# 各モデルについて、"Inter"（内挿）と"Extra"（外挿）の2系列を作成
series_data = {}
error_data = {}
models = list(csv_files.keys())
for model in models:
    series_data[f"{model}_Inter"] = []
    series_data[f"{model}_Extra"] = []
    error_data[f"{model}_Inter"] = []
    error_data[f"{model}_Extra"] = []

# --- 各targetごとに正規化処理（対象内の最大RMSE値で割る） ---
for target in targets:
    temp_data = (
        {}
    )  # 各モデルの値を一時保存（内挿、内挿エラー、外挿、外挿エラー）
    values = []  # 正規化のため、全モデルのRMSE値を収集
    for model in models:
        df = dfs[model]
        target_data = df[df['target'] == target]
        if not target_data.empty:
            inter = target_data['val_rmse_mean'].values[0]
            extra = target_data['test_rmse_mean'].values[0]
            # モデルSRはエラーバーなし（=0）とする
            if model == 'SR':
                inter_err = 0
                extra_err = 0
            else:
                inter_err = np.sqrt(target_data['val_rmse_variance'].values[0])
                extra_err = np.sqrt(
                    target_data['test_rmse_variance'].values[0]
                )
            temp_data[model] = (inter, inter_err, extra, extra_err)
            values.extend([inter, extra])
        else:
            temp_data[model] = None
    # そのtargetにおける最大RMSE値で正規化（万一0の場合は1に）
    if values:
        norm_factor = max(values)
        if norm_factor == 0:
            norm_factor = 1
    else:
        norm_factor = 1

    # 各モデルごとに正規化済み値を各系列に追加
    for model in models:
        if temp_data[model] is not None:
            inter, inter_err, extra, extra_err = temp_data[model]
            norm_inter = inter / norm_factor
            norm_extra = extra / norm_factor
            norm_inter_err = inter_err / norm_factor
            norm_extra_err = extra_err / norm_factor
        else:
            norm_inter = np.nan
            norm_extra = np.nan
            norm_inter_err = np.nan
            norm_extra_err = np.nan

        series_data[f"{model}_Inter"].append(norm_inter)
        series_data[f"{model}_Extra"].append(norm_extra)
        error_data[f"{model}_Inter"].append(norm_inter_err)
        error_data[f"{model}_Extra"].append(norm_extra_err)

# --- グラフ描画（内挿と外挿を別々のグラフに） ---
pdf_path = os.path.join(save_dir, "rmse_normalized_separate.pdf")
with PdfPages(pdf_path) as pdf:
    # 【内挿（Interpolation）のグラフ】
    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(len(targets))  # 各targetのx軸位置
    n_models = len(models)
    bar_width = 0.2
    # 各target内に3本の棒を配置するため、中央に集めるオフセットを設定
    offsets = [(i - (n_models - 1) / 2) * bar_width for i in range(n_models)]

    # モデルごとの色設定
    model_colors = {'LR': 'red', 'NN': 'blue', 'SR': 'green'}

    for i, model in enumerate(models):
        pos = x + offsets[i]
        ax.bar(
            pos,
            series_data[f"{model}_Inter"],
            width=bar_width,
            yerr=error_data[f"{model}_Inter"],
            label=model,
            color=model_colors.get(model, 'gray'),
            capsize=5,
        )

    ax.set_xlabel('Target', fontsize=20)
    ax.set_ylabel('Normalized RMSE', fontsize=20)
    ax.set_title('Normalized RMSE Comparison - Interpolation', fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # 【外挿（Extrapolation）のグラフ】
    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(len(targets))
    offsets = [(i - (n_models - 1) / 2) * bar_width for i in range(n_models)]

    for i, model in enumerate(models):
        pos = x + offsets[i]
        ax.bar(
            pos,
            series_data[f"{model}_Extra"],
            width=bar_width,
            yerr=error_data[f"{model}_Extra"],
            label=model,
            color=model_colors.get(model, 'gray'),
            capsize=5,
        )

    ax.set_xlabel('Target', fontsize=20)
    ax.set_ylabel('Normalized RMSE', fontsize=20)
    ax.set_title('Normalized RMSE Comparison - Extrapolation', fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved separate graphs in: {pdf_path}")
