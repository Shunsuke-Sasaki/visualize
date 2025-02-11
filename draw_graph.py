import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# CSVファイルのパスを辞書で定義（モデル名とファイルパスを対応付け）
csv_files = {
    'LR': '/Users/sshunsuke/Downloads/rmse_statistics_linear.csv',
    'NN': '/Users/sshunsuke/Downloads/rmse_statistics_nn.csv',
    'SR': '/Users/sshunsuke/Downloads/rmse_sr.csv',
}

# 対象とするtargetを指定
target = 'Ebd'

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
            # Varianceが存在する場合
            data['Interpolation RMSE Variance'].append(
                target_data['val_rmse_variance'].values[0]
            )
            data['Extrapolation RMSE Variance'].append(
                target_data['test_rmse_variance'].values[0]
            )
        else:
            # SRの場合、Varianceは0として扱う
            data['Interpolation RMSE Variance'].append(0)
            data['Extrapolation RMSE Variance'].append(0)
    else:
        print(f"Warning: Target {target} not found in {model}'s CSV file.")

# DataFrameを作成
df = pd.DataFrame(data)
print(df)

# 棒グラフの描画
fig, ax = plt.subplots(figsize=(10, 6))

# 棒グラフの幅
bar_width = 0.2
index = np.arange(len(df))

# Interpolation MSEの棒グラフ（誤差バーを追加）
for i in range(len(df)):
    if df['Model'][i] == 'SR':
        # SRモデルは誤差バーなし
        ax.bar(
            index[i] - bar_width / 2,
            df['Interpolation RMSE'][i],
            bar_width,
            label='Interpolation RMSE' if i == 0 else "",
            color='red',
        )
        ax.bar(
            index[i] + bar_width / 2,
            df['Extrapolation RMSE'][i],
            bar_width,
            label='Extrapolation RMSE' if i == 0 else "",
            color='blue',
        )
    else:
        # その他のモデルは誤差バーあり
        ax.bar(
            index[i] - bar_width / 2,
            df['Interpolation RMSE'][i],
            bar_width,
            yerr=np.sqrt(df['Interpolation RMSE Variance'][i]),
            label='Interpolation RMSE' if i == 0 else "",
            color='red',
            capsize=5,
        )
        ax.bar(
            index[i] + bar_width / 2,
            df['Extrapolation RMSE'][i],
            bar_width,
            yerr=np.sqrt(df['Extrapolation RMSE Variance'][i]),
            label='Extrapolation RMSE' if i == 0 else "",
            color='blue',
            capsize=5,
        )

# ラベル設定
ax.set_xlabel('Model', fontsize=15)
ax.set_ylabel('RMSE', fontsize=15)
ax.set_title(
    'RMSE Comparison',
    fontsize=14,
)

# モデル名の表示
ax.set_xticks(index)
ax.set_xticklabels(df['Model'], fontsize=13)

# 凡例を追加
ax.legend()

# グリッドの表示
ax.grid(True, linestyle='--', alpha=0.7)

# グラフの表示
plt.tight_layout()
plt.show()
