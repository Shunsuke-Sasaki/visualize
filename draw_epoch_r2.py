import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込みます
file_path = '/Users/sshunsuke/Downloads/results_epochs_evaluation.csv'  # 実際のファイルパスに置き換えてください
data = pd.read_csv(file_path)

# 各 target と epochs において fold の平均を計算します
grouped = data.groupby(['target', 'epochs']).mean().reset_index()

# 各 target についてプロットを作成
plt.figure(figsize=(12, 8))

targets = grouped['target'].unique()
for target in targets:
    subset = grouped[grouped['target'] == target]
    plt.plot(
        subset['epochs'],
        subset['val_r2'],
        label=f'Validation R² (Target: {target})',
        marker='o',
    )
    plt.plot(
        subset['epochs'],
        subset['test_r2'],
        label=f'Test R² (Target: {target})',
        linestyle='--',
        marker='o',
    )

# グラフの設定
plt.xscale('log')  # エポック数を対数スケールに
plt.xlabel('Epochs (log scale)')
plt.ylabel('R²')
plt.title('R² vs Epochs for All Targets')
plt.legend(loc='best')
plt.grid(True)

# グラフを表示
plt.show()
