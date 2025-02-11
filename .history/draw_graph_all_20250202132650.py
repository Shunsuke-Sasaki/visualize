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

# **1つのPDFファイルにまとめて保存**
pdf_path = os.path.join(save_dir, "rmse_all.pdf")
with PdfPages(pdf_path) as pdf:
    # 各targetについてグラフを描画し保存
    for target in targets:
        # データを格納するリスト
        data = {
            'Model': [],
            'Extrapolation RMSE': [],
            'Interpolation RMSE': [],
            'Extrapolation RMSE Variance': [],
            'Interpolation RMSE Variance': [],
        }

        # CSVファイルを読み込んでデータを抽出
        for model, file_path in csv_files.items():
            df = pd.read_csv(file_path)
            target_data = df[df['target'] == target]
            if not target_data.empty:
                data['Model'].append(model)
                data['Interpolation RMSE'].append(
                    target_data['val_rmse_mean'].values[0]
                )
                data['Extrapolation RMSE'].append(
                    target_data['test_rmse_mean'].values[0]
                )
                if model != 'SR':
                    data['Interpolation RMSE Variance'].append(
                        target_data['val_rmse_variance'].values[0]
                    )
                    data['Extrapolation RMSE Variance'].append(
                        target_data['test_rmse_variance'].values[0]
                    )
                else:
                    data['Interpolation RMSE Variance'].append(0)
                    data['Extrapolation RMSE Variance'].append(0)
            else:
                print(
                    f"Warning: Target {target} not found in {model}'s CSV file."
                )

        # DataFrameを作成
        df = pd.DataFrame(data)
        print(df)

        # グラフの描画
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.2
        index = np.arange(len(df))

        for i in range(len(df)):
            if df['Model'][i] == 'SR':
                ax.bar(
                    index[i] - bar_width / 2,
                    df['Interpolation RMSE'][i],
                    bar_width,
                    color='red',
                )
                ax.bar(
                    index[i] + bar_width / 2,
                    df['Extrapolation RMSE'][i],
                    bar_width,
                    color='blue',
                )
            else:
                ax.bar(
                    index[i] - bar_width / 2,
                    df['Interpolation RMSE'][i],
                    bar_width,
                    yerr=np.sqrt(df['Interpolation RMSE Variance'][i]),
                    color='red',
                    capsize=5,
                )
                ax.bar(
                    index[i] + bar_width / 2,
                    df['Extrapolation RMSE'][i],
                    bar_width,
                    yerr=np.sqrt(df['Extrapolation RMSE Variance'][i]),
                    color='blue',
                    capsize=5,
                )

        ax.set_xlabel('Model', fontsize=24)
        ax.set_ylabel(f'RMSE [{rmse_units[target]}]', fontsize=24)
        ax.set_title(f'RMSE Comparison for {target}', fontsize=28)
        ax.set_xticks(index)
        ax.set_xticklabels(df['Model'], fontsize=22)
        ax.set_yticklabels(ax.get_yticks(), fontsize=22)
        ax.legend(['Interpolation RMSE', 'Extrapolation RMSE'], fontsize=22)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.3g}')
        )

        plt.tight_layout()

        # **PDFに追加**
        pdf.savefig(fig)  # PDFに現在のグラフを保存
        plt.close(fig)  # メモリ解放

    print(f"Saved all graphs in: {pdf_path}")
