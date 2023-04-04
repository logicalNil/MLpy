import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('iris.csv')

print("Number of columns present:-",
      len(df.columns))
print("Number of rows present:-", 
      len(df.index))
print(df.dtypes)

print(df.head(5))

item_cou = df["species"].value_counts()
print(item_cou)

print(df["species"].unique())

# remove versicolor from iris dataset
versi = df[df["species"] != "versicolor"]
print(versi)

# remove species column from versi
new = versi.drop("species", axis=1)
print(new)

# decision tree construction
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split