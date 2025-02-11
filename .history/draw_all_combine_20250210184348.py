import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

# --- CSVファイルのパス定義 ---
csv_files = {
    'LR': '/Users/sshunsuke/Downloads/rmse_statistics_LR_normalized.csv',
    'NN': '/Users/sshunsuke/Downloads/rmse_statistics_nn_norm.csv',
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
# 内挿（Inter）と外挿（Extra）をそれぞれ独立に扱います。
series_data = {}
error_data = {}
models = list(csv_files.keys())
for model in models:
    series_data[f"{model}_Inter"] = []  # 内挿の正規化済みRMSE
    series_data[f"{model}_Extra"] = []  # 外挿の正規化済みRMSE
    error_data[f"{model}_Inter"] = []  # 内挿のエラーデータ
    error_data[f"{model}_Extra"] = []  # 外挿のエラーデータ

# --- 各targetごとに内挿と外挿を個別に正規化 ---
for target in targets:
    temp_data = (
        {}
    )  # 各モデルの元の値を一時保存 (inter, inter_err, extra, extra_err)
    inter_values = []  # 内挿のRMSE値を集めるリスト
    extra_values = []  # 外挿のRMSE値を集めるリスト

    for model in models:
        df = dfs[model]
        target_data = df[df['target'] == target]
        if not target_data.empty:
            # 内挿値
            inter = target_data['val_rmse_mean'].values[0]
            # 外挿値
            extra = target_data['test_rmse_mean'].values[0]
            # モデルSRはエラーバーなし（0）とする
            if model == 'SR':
                inter_err = 0
                extra_err = 0
            else:
                inter_err = np.sqrt(target_data['val_rmse_variance'].values[0])
                extra_err = np.sqrt(
                    target_data['test_rmse_variance'].values[0]
                )
            temp_data[model] = (inter, inter_err, extra, extra_err)
            inter_values.append(inter)
            extra_values.append(extra)
        else:
            temp_data[model] = None

    # 内挿と外挿で正規化に用いる各target内の最大値を個別に算出
    if inter_values:
        norm_factor_inter = max(inter_values)
        if norm_factor_inter == 0:
            norm_factor_inter = 1
    else:
        norm_factor_inter = 1

    if extra_values:
        norm_factor_extra = max(extra_values)
        if norm_factor_extra == 0:
            norm_factor_extra = 1
    else:
        norm_factor_extra = 1

    # 各モデルごとに正規化した値を格納
    for model in models:
        if temp_data[model] is not None:
            inter, inter_err, extra, extra_err = temp_data[model]
            norm_inter = inter / norm_factor_inter
            norm_extra = extra / norm_factor_extra
            norm_inter_err = inter_err / norm_factor_inter
            norm_extra_err = extra_err / norm_factor_extra
        else:
            norm_inter = np.nan
            norm_extra = np.nan
            norm_inter_err = np.nan
            norm_extra_err = np.nan

        series_data[f"{model}_Inter"].append(norm_inter)
        series_data[f"{model}_Extra"].append(norm_extra)
        error_data[f"{model}_Inter"].append(norm_inter_err)
        error_data[f"{model}_Extra"].append(norm_extra_err)

# --- グラフ描画 & PDF出力 ---
pdf_path = os.path.join(save_dir, "rmse_normalized_separate_norm.pdf")
with PdfPages(pdf_path) as pdf:

    # 【内挿（Interpolation）のグラフ】
    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(len(targets))  # 各targetのx軸位置
    n_models = len(models)
    bar_width = 0.2
    # 各target内にモデルごとの棒を中央に配置するためのオフセットを設定
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
    ax.set_ylabel('Normalized RMSE (Interpolation)', fontsize=20)
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
    ax.set_ylabel('Normalized RMSE (Extrapolation)', fontsize=20)
    ax.set_title('Normalized RMSE Comparison - Extrapolation', fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved separate normalized graphs in: {pdf_path}")
