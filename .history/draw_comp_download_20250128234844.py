import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ターゲット列のリスト
target_columns = [
    'dGs',
    'Ebd',
    'log10(lifetime)',
    'logD',
    'logP',
    'logS',
    'pKaA',
    'pKaB',
    'RI',
    'Tb',
    'Tm',
]

for target in target_columns:
    # CSVファイルを読み込む（ファイル名を適宜変更してください）
    file_path = f"/Users/sshunsuke/Downloads/error_results_{target}.csv"  # CSVファイルのパス
    df = pd.read_csv(file_path)

    # Lossの平方根を計算
    df["Sqrt_Loss"] = np.sqrt(df["Loss"])
    df = df[df["Complexity"] >= 10]

    # グラフをプロット
    plt.figure(figsize=(10, 6))
    plt.plot(
        df["Complexity"],
        df["Sqrt_Loss"],
        marker="o",
        label="interpolation RMSE",
        markersize=5,
    )
    plt.plot(
        df["Complexity"],
        df["Range2_RMSE"],
        marker="o",
        label="extrapolation RMSE",
        markersize=5,
    )

    # ラベルとタイトルを追加
    plt.xlabel("Complexity")
    plt.ylabel("RMSE")
    plt.title(
        f"Complexity vs interpolation and extrapolation RMSE of {target}"
    )
    plt.legend()
    plt.grid()

    # グラフを保存
    output_path = f"./Complexity_vs_RMSE_{target}.png"  # 保存先のパス
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Graph saved as: {output_path}")

    # グラフを表示
    plt.show()
