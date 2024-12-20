from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
#save raw iris
df.to_csv('./data/raw/iris.csv', index=False)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv('data/prepare/train.csv', index=False)
test_df.to_csv('data/prepare/test.csv', index=False)
