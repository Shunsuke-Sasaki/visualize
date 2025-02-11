import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込みます
file_path = 'your_file_path.csv'  # 実際のファイルパスに置き換えてください
data = pd.read_csv(file_path)

# 各 target と epochs において fold の平均を計算します
grouped = data.groupby(['target', 'epochs']).mean().reset_index()

# 各 target についてプロットを作成
targets = grouped['target'].unique()

for target in targets:
    subset = grouped[grouped['target'] == target]
    plt.figure(figsize=(10, 6))
    plt.plot(
        subset['epochs'],
        subset['val_rmse'],
        label='Validation RMSE',
        marker='o',
    )
    plt.plot(
        subset['epochs'], subset['test_rmse'], label='Test RMSE', marker='o'
    )
    plt.xscale('log')  # エポック数を対数スケールに
    plt.xlabel('Epochs (log scale)')
    plt.ylabel('RMSE')
    plt.title(f'RMSE vs Epochs for Target: {target}')
    plt.legend()
    plt.grid(True)
    plt.show()
