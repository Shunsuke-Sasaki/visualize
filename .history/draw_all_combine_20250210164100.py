import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages  # PDF出力用

# CSVファイルのパスを辞書で定義（モデル名とファイルパスの対応）
csv_files = {
    'LR': '/Users/sshunsuke/Downloads/rmse_statistics_linear.csv',
    'NN': '/Users/sshunsuke/Downloads/rmse_statistics_nn.csv',
    'SR': '/Users/sshunsuke/Downloads/rmse_sr.csv',
}

# RMSEの単位（正規化後は単位が失われるため、元の単位は表示しない）
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

# 対象とする予測対象のリスト
targets = list(rmse_units.keys())

# 保存先フォルダの指定
save_dir = "/Users/sshunsuke/Downloads/"
os.makedirs(save_dir, exist_ok=True)  # フォルダがなければ作成

# 各CSVファイルをあらかじめ読み込む
dfs = {}
for model, file_path in csv_files.items():
    dfs[model] = pd.read_csv(file_path)

# --- 各モデル・RMSEタイプごとの系列を格納する辞書の初期化 ---
# キーは "モデル名_Inter" および "モデル名_Extra" としておく
series_data = {}
error_data = {}
models = list(csv_files.keys())
for model in models:
    series_data[f"{model}_Inter"] = []
    series_data[f"{model}_Extra"] = []
    error_data[f"{model}_Inter"] = []
    error_data[f"{model}_Extra"] = []

# --- 各targetごとに正規化して値を計算 ---
# 各target内で利用可能なモデルのInterpolation・ExtrapolationのRMSEの最大値で割って正規化する
for target in targets:
    temp_data = (
        {}
    )  # 各モデルごとのデータを一時格納（タプル： (inter, inter_err, extra, extra_err)）
    values = (
        []
    )  # 正規化のため、全モデルのRMSE値（Interpolation, Extrapolation）を集める
    for model in models:
        df = dfs[model]
        target_data = df[df['target'] == target]
        if not target_data.empty:
            inter = target_data['val_rmse_mean'].values[0]
            extra = target_data['test_rmse_mean'].values[0]
            # モデルSRはエラーバーがない（=0）とする
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

    # そのtargetにおける最大RMSE値（利用可能な値のみ）を正規化のスケールとする
    if values:
        norm_factor = max(values)
        if norm_factor == 0:
            norm_factor = 1  # 万一0の場合は1にする（ゼロ除算回避）
    else:
        norm_factor = 1

    # 各モデルごとに正規化した値をシリーズに追加（データが無い場合はnp.nan）
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

# --- プロット ---
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(targets))  # x軸の位置（各対象）

# モデルごとの色設定（例）
model_colors = {'LR': 'red', 'NN': 'blue', 'SR': 'green'}

# RMSEタイプごとのマーカーとラインスタイルの設定
rmse_styles = {
    'Inter': {'marker': 'o', 'linestyle': '-'},
    'Extra': {'marker': 's', 'linestyle': '--'},
}

# 各モデル・RMSEタイプの系列をエラーバー付きでプロット
for model in models:
    for rmse_type in ['Inter', 'Extra']:
        key = f"{model}_{rmse_type}"
        ax.errorbar(
            x,
            series_data[key],
            yerr=error_data[key],
            label=f"{model} {'Interpolation' if rmse_type=='Inter' else 'Extrapolation'}",
            color=model_colors[model],
            marker=rmse_styles[rmse_type]['marker'],
            linestyle=rmse_styles[rmse_type]['linestyle'],
            capsize=5,
        )

ax.set_xlabel('Target', fontsize=20)
ax.set_ylabel('Normalized RMSE', fontsize=20)
ax.set_title('Normalized RMSE Comparison Across Targets', fontsize=24)
ax.set_xticks(x)
ax.set_xticklabels(targets, rotation=45, fontsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# --- PDFに保存 ---
pdf_path = os.path.join(save_dir, "rmse_normalized.pdf")
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig)
plt.show()
plt.close(fig)

print(f"Saved normalized RMSE graph in: {pdf_path}")
