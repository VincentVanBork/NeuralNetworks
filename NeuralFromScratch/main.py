import numpy as np
import random
import pandas as pd

iris_data = pd.read_csv('./NeuralFromScratch/iris.csv', delimiter=',')
df = iris_data[['sepal_length','sepal_width','petal_length','petal_width']]
# iris_data_norm=(df-df.min())/(df.max()-df.min())
iris_data_norm = df/df.max()
# print(iris_data_norm)

