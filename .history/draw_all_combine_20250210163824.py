import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages  # PDF出力用

# CSVファイルのパスを辞書で定義（モデル名とファイルパスを対応付け）
csv_files = {
    'LR': '/Users/sshunsuke/Downloads/rmse_statistics_linear.csv',
    'NN': '/Users/sshunsuke/Downloads/rmse_statistics_nn.csv',
    'SR': '/Users/sshunsuke/Downloads/rmse_sr.csv',
}

# RMSEの単位（参考用）
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

# 対象とする targets のリスト
targets = list(rmse_units.keys())

# **保存先フォルダの指定**
save_dir = "/Users/sshunsuke/Downloads/"
os.makedirs(save_dir, exist_ok=True)  # フォルダがない場合は作成

# **PDFに保存**
pdf_path = os.path.join(save_dir, "rmse_overview_normalized.pdf")

with PdfPages(pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(16, 8))
    bar_width = 0.12  # 各バーの幅
    index = np.arange(len(targets))  # X 軸位置

    # データ格納用
    rmse_values = {
        model: {
            'Interpolation': [],
            'Extrapolation': [],
            'Interpolation Variance': [],
            'Extrapolation Variance': [],
        }
        for model in csv_files
    }

    # CSVファイルからデータを取得
    for model, file_path in csv_files.items():
        df = pd.read_csv(file_path)
        for target in targets:
            target_data = df[df['target'] == target]
            if not target_data.empty:
                rmse_values[model]['Interpolation'].append(
                    target_data['val_rmse_mean'].values[0]
                )
                rmse_values[model]['Extrapolation'].append(
                    target_data['test_rmse_mean'].values[0]
                )
                if model != 'SR':
                    rmse_values[model]['Interpolation Variance'].append(
                        np.sqrt(target_data['val_rmse_variance'].values[0])
                    )
                    rmse_values[model]['Extrapolation Variance'].append(
                        np.sqrt(target_data['test_rmse_variance'].values[0])
                    )
                else:
                    rmse_values[model]['Interpolation Variance'].append(0)
                    rmse_values[model]['Extrapolation Variance'].append(0)
            else:
                print(
                    f"Warning: Target {target} not found in {model}'s CSV file."
                )
                rmse_values[model]['Interpolation'].append(0)
                rmse_values[model]['Extrapolation'].append(0)
                rmse_values[model]['Interpolation Variance'].append(0)
                rmse_values[model]['Extrapolation Variance'].append(0)

    # **正規化処理（Min-Max Normalization）**
    for key in [
        'Interpolation',
        'Extrapolation',
        'Interpolation Variance',
        'Extrapolation Variance',
    ]:
        all_values = np.array(
            [rmse_values[model][key] for model in csv_files]
        )  # shape (3, len(targets))
        min_vals = np.min(all_values, axis=0)  # 最小値（targetごと）
        max_vals = np.max(all_values, axis=0)  # 最大値（targetごと）
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # すべての値が同じ場合、ゼロ除算を防ぐ

        # 各モデルのデータを正規化
        for model in csv_files:
            rmse_values[model][key] = (
                np.array(rmse_values[model][key]) - min_vals
            ) / range_vals

    # **グラフ描画**
    colors = {'LR': 'red', 'NN': 'blue', 'SR': 'green'}
    model_positions = {
        'LR': -bar_width,
        'NN': 0,
        'SR': bar_width,
    }  # モデルごとのオフセット

    for model, offset in model_positions.items():
        ax.bar(
            index + offset - bar_width / 2,
            rmse_values[model]['Interpolation'],
            bar_width,
            yerr=rmse_values[model]['Interpolation Variance'],
            label=f'{model} Interpolation',
            color=colors[model],
            alpha=0.7,
            capsize=5,
        )
        ax.bar(
            index + offset + bar_width / 2,
            rmse_values[model]['Extrapolation'],
            bar_width,
            yerr=rmse_values[model]['Extrapolation Variance'],
            label=f'{model} Extrapolation',
            color=colors[model],
            hatch='/',
            alpha=0.7,
            capsize=5,
        )

    # 軸ラベルとタイトル
    ax.set_xlabel('Prediction Target', fontsize=18)
    ax.set_ylabel('Normalized RMSE (0 to 1)', fontsize=18)
    ax.set_title(
        'Normalized RMSE and Variance Comparison Across Targets', fontsize=22
    )

    # X 軸を調整して 6 本のバーが対象ごとにまとまるようにする
    tick_positions = index  # 各 target の中心位置
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=14)

    ax.legend(
        fontsize=14, loc='upper left', bbox_to_anchor=(1, 1)
    )  # 凡例を右上に配置
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # PDFに保存
    pdf.savefig(fig)
    plt.show()

    print(f"Saved normalized overview graph in: {pdf_path}")
