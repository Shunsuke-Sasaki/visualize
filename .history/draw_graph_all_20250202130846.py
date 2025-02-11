import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# CSVファイルのパスを辞書で定義（モデル名とファイルパスを対応付け）
csv_files = {
    'LR': '/Users/sshunsuke/Downloads/rmse_statistics_linear.csv',
    'NN': '/Users/sshunsuke/Downloads/rmse_statistics_nn.csv',
    'SR': '/Users/sshunsuke/Downloads/rmse_sr.csv',
}
# 対象とする targets のリスト
targets = [
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
# 例として複数のtargetを指定

# 各targetについてグラフを描画
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

    # グラフの描画
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    index = np.arange(len(df))

    # InterpolationとExtrapolationの棒グラフを描画
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

    ax.set_xlabel('Model', fontsize=20)
    ax.set_ylabel('RMSE', fontsize=20)
    ax.set_title(f'RMSE Comparison for {target}', fontsize=0)
    ax.set_xticks(index)
    ax.set_xticklabels(df['Model'], fontsize=13)
    ax.legend(['Interpolation RMSE', 'Extrapolation RMSE'])
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
