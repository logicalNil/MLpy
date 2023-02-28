import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas
import numpy
import sys



url = "/workspaces/MLpy/iris.csv"
dataset = pandas.read_csv(url)

print(dataset.head(20))

plt.figure()
fig, ax = plt.subplots(1, 2, figsize=(17, 9))
dataset.plot(x="sepal_length", y="sepal_width", kind="scatter",
             ax=ax[0], sharex=False, sharey=False, label="sepal", color='r')
dataset.plot(x="petal_length", y="petal_width", kind="scatter",
             ax=ax[1], sharex=False, sharey=False, label="petal", color='b')
ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()
