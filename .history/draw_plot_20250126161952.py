import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSVファイルを読み込む（ファイル名を適宜変更してください）
file_path = (
    "/Users/sshunsuke/Downloads/error_results_Tm.csv"  # CSVファイルのパス
)
df = pd.read_csv(file_path)

# Lossの平方根を計算
df["Sqrt_Loss"] = np.sqrt(df["Loss"])

# グラフをプロット
plt.figure(figsize=(10, 6))
plt.plot(
    df["Complexity"], df["Sqrt_Loss"], marker="o", label="interpolation RMSE"
)
plt.plot(
    df["Complexity"],
    df["Range2_RMSE"],
    marker="o",
    label="extrapolation RMSE",
)

# ラベルとタイトルを追加
plt.xlabel("Complexity")
plt.ylabel("Values")
plt.title("Complexity vs √Loss and Range2_RMSE")
plt.legend()
plt.grid()

# グラフを表示
plt.show()
