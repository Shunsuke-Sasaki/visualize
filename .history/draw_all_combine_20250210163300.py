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

# RMSEの単位
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
pdf_path = os.path.join(save_dir, "rmse_overview.pdf")

with PdfPages(pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.2
    index = np.arange(len(targets))

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

    # 各モデルの RMSE をプロット
    colors = {'LR': 'red', 'NN': 'blue', 'SR': 'green'}
    for i, (model, values) in enumerate(rmse_values.items()):
        ax.bar(
            index + i * bar_width - bar_width,
            values['Interpolation'],
            bar_width,
            yerr=values['Interpolation Variance'],
            label=f'{model} Interpolation',
            color=colors[model],
            alpha=0.7,
            capsize=5,
        )
        ax.bar(
            index + i * bar_width,
            values['Extrapolation'],
            bar_width,
            yerr=values['Extrapolation Variance'],
            label=f'{model} Extrapolation',
            color=colors[model],
            hatch='/',
            alpha=0.7,
            capsize=5,
        )

    # 軸ラベルとタイトル
    ax.set_xlabel('Prediction Target', fontsize=18)
    ax.set_ylabel('RMSE', fontsize=18)
    ax.set_title('RMSE Comparison Across Targets', fontsize=22)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # PDFに保存
    pdf.savefig(fig)
    plt.show()

    print(f"Saved overview graph in: {pdf_path}")
