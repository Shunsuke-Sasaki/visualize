import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

# --- CSVファイルのパス定義 ---
csv_files = {
    'LR': '/Users/sshunsuke/Downloads/rmse_statistics_lr_norm_0.4.csv',
    'NN': '/Users/sshunsuke/Downloads/rmse_statistics_nn_norm_0.4.csv',
    #'SR': '/Users/sshunsuke/Downloads/rmse_sr.csv',
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

# --- 内挿（Inter）と外挿（Extra）の正規化済み値およびエラーデータを格納する辞書の初期化 ---
# ここでは各モデルごとに、内挿と外挿の2系列を用意します。
series_data = {}
error_data = {}
models = list(csv_files.keys())
for model in models:
    series_data[f"{model}_Inter"] = []  # 内挿の正規化済みRMSE
    series_data[f"{model}_Extra"] = []  # 外挿の正規化済みRMSE
    error_data[f"{model}_Inter"] = []  # 内挿のエラーデータ
    error_data[f"{model}_Extra"] = []  # 外挿のエラーデータ

# --- 各targetごとに、LRモデルの値を基準にして内挿と外挿を個別に正規化 ---
for target in targets:
    temp_data = (
        {}
    )  # 各モデルの元の値を一時保存 (inter, inter_err, extra, extra_err)
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
        else:
            temp_data[model] = None

    # 正規化の基準として、LRモデルの値を使用し、内挿と外挿で同じ係数を使う
    if temp_data['LR'] is not None:
        lr_inter, lr_inter_err, lr_extra, lr_extra_err = temp_data['LR']
        # LRモデルの内挿値と外挿値のうち大きい方を基準値とする
        norm_factor = (
            max(lr_inter, lr_extra) if max(lr_inter, lr_extra) != 0 else 1
        )
    else:
        norm_factor = 1

    # 各モデルごとに、LRを基準とした正規化値を算出（内挿も外挿も同じ基準で割る）
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

# --- グラフ描画時に内挿・外挿で同一の y 軸スケールにするため、全体の最大値を取得 ---
max_inter_values = []
for model in models:
    max_val = np.nanmax(series_data[f"{model}_Inter"])
    max_inter_values.append(max_val)
global_max_inter = max(max_inter_values)

max_extra_values = []
for model in models:
    max_val = np.nanmax(series_data[f"{model}_Extra"])
    max_extra_values.append(max_val)
global_max_extra = max(max_extra_values)

global_max = max(global_max_inter, global_max_extra)
y_limit = global_max * 1.1  # 10%余裕を持たせる

# --- グラフ描画 & PDF出力 ---
pdf_path = os.path.join(save_dir, "rmse_normalized_lr_based.pdf")
with PdfPages(pdf_path) as pdf:

    # 【内挿（Interpolation）のグラフ】
    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(len(targets))  # 各targetのx軸位置
    n_models = len(models)
    bar_width = 0.2
    # 各target内にモデルごとの棒を中央に配置するためのオフセット
    offsets = [(i - (n_models - 1) / 2) * bar_width for i in range(n_models)]
    # 内挿用の色設定（LR: red, NN: blue, SR: green）
    colors_inter = {'LR': 'red', 'NN': 'blue', 'SR': 'green'}

    for i, model in enumerate(models):
        pos = x + offsets[i]
        ax.bar(
            pos,
            series_data[f"{model}_Inter"],
            width=bar_width,
            yerr=error_data[f"{model}_Inter"],
            label=model,
            color=colors_inter.get(model, 'gray'),
            capsize=5,
        )

    ax.set_xlabel('Target', fontsize=20)
    ax.set_ylabel('Normalized RMSE (Interpolation)', fontsize=20)
    ax.set_title(
        'Normalized RMSE Comparison - Interpolation (LR-based)', fontsize=24
    )
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, y_limit)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # 【外挿（Extrapolation）のグラフ】
    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(len(targets))
    offsets = [(i - (n_models - 1) / 2) * bar_width for i in range(n_models)]
    # 外挿用の色設定（LR: red, NN: blue, SR: green）
    colors_extra = {'LR': 'red', 'NN': 'blue', 'SR': 'green'}
    for i, model in enumerate(models):
        pos = x + offsets[i]
        ax.bar(
            pos,
            series_data[f"{model}_Extra"],
            width=bar_width,
            yerr=error_data[f"{model}_Extra"],
            label=model,
            color=colors_extra.get(model, 'gray'),
            capsize=5,
        )

    ax.set_xlabel('Target', fontsize=20)
    ax.set_ylabel('Normalized RMSE (Extrapolation)', fontsize=20)
    ax.set_title(
        'Normalized RMSE Comparison - Extrapolation (LR-based)', fontsize=24
    )
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, y_limit)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved LR-based normalized graphs in: {pdf_path}")
