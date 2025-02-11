import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages  # PDF出力用

# RMSEの単位を定義
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

# CSVファイルのパス
file_path = '/Users/sshunsuke/Downloads/results_epochs_evaluation.csv'

# **保存先フォルダを指定**
save_dir = "/Users/sshunsuke/Downloads/rmse_epochs_graphs"
os.makedirs(save_dir, exist_ok=True)  # フォルダがない場合は作成

# **1つのPDFファイルにまとめて保存**
pdf_path = os.path.join(save_dir, "rmse_epochs_all_targets.pdf")
with PdfPages(pdf_path) as pdf:
    try:
        # CSVファイルを読み込み
        data = pd.read_csv(file_path)

        # 各 target と epochs において fold の平均を計算
        grouped = data.groupby(['target', 'epochs']).mean().reset_index()

        # 各 target についてプロットを作成
        targets = grouped['target'].unique()

        for target in targets:
            subset = grouped[grouped['target'] == target]

            # グラフの描画
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(
                subset['epochs'],
                subset['val_rmse'],
                label='Validation RMSE',
                marker='o',
                markersize=8,
            )
            ax.plot(
                subset['epochs'],
                subset['test_rmse'],
                label='Test RMSE',
                marker='o',
                markersize=8,
            )

            # x軸を対数スケールに設定
            ax.set_xscale('log')

            # ラベルとタイトル（フォントサイズを統一）
            ax.set_xlabel("Epochs (log scale)", fontsize=24)
            ax.set_ylabel(
                f"RMSE [{rmse_units.get(target, '')}]", fontsize=24
            )  # 単位を追加
            ax.set_title(
                f"RMSE vs NN Training Epochs for {target}", fontsize=28
            )

            # 軸の目盛りのフォントサイズを調整
            ax.tick_params(axis="both", labelsize=22)

            # y軸の目盛りを有効数字3桁で表示
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{x:.3g}')
            )

            # 凡例とグリッドを設定
            ax.legend(fontsize=22)
            ax.grid(True, linestyle="--", alpha=0.7)

            plt.tight_layout()

            # **PDFに追加**
            pdf.savefig(fig)  # PDFに現在のグラフを保存
            plt.close(fig)  # メモリ解放

        print(f"Saved all graphs in: {pdf_path}")

    except FileNotFoundError:
        print(f"Warning: File not found - {file_path}")
