import pandas as pd
from sklearn.datasets import load_wine

wine_dataset = load_wine()

X = pd.DataFrame(wine_dataset.data, columns=wine_dataset.feature_names)

y = pd.DataFrame(wine_dataset.target, columns=['target'])

print("カラムの数は:", X.shape)

print("要素の種類と出現数を表示して:", y['target'].value_counts())

# くっつけて十行表示
combined = pd.concat([X, y], axis=1)
print("最初の十行を表示します。:")
print(combined.head(10))
