import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

# 対象のターゲット変数
target_columns = list(rmse_units.keys())

# **保存先フォルダを指定**
save_dir = "/Users/sshunsuke/Downloads/rmse_complexity_graphs"
os.makedirs(save_dir, exist_ok=True)  # フォルダがない場合は作成

# **1つのPDFファイルにまとめて保存**
pdf_path = os.path.join(save_dir, "rmse_complexity_all.pdf")
with PdfPages(pdf_path) as pdf:
    # 各ターゲットについて処理
    for target in target_columns:
        # CSVファイルのパス
        file_path = f"/Users/sshunsuke/Downloads/error_results_{target}.csv"

        try:
            # CSVを読み込む
            df = pd.read_csv(file_path)

            # Lossの平方根を計算し、新しい列を追加
            df["Sqrt_Loss"] = np.sqrt(df["Loss"])

            # Complexityが10以上のデータのみを使用
            df = df[df["Complexity"] >= 10]

            # グラフの描画
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(
                df["Complexity"],
                df["Sqrt_Loss"],
                marker="o",
                label="Interpolation RMSE",
                markersize=8,
            )
            ax.plot(
                df["Complexity"],
                df["Range2_RMSE"],
                marker="o",
                label="Extrapolation RMSE",
                markersize=8,
            )

            # ラベルとタイトル（フォントサイズを統一）
            ax.set_xlabel("Complexity", fontsize=24)
            ax.set_ylabel(
                f"RMSE [{rmse_units[target]}]", fontsize=24
            )  # 単位を追加
            ax.set_title(
                f"Complexity RMSE of {target}",
                fontsize=28,
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

        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")

    print(f"Saved all graphs in: {pdf_path}")
