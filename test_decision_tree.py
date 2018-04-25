from decision_tree import decision_tree
import pandas as pd
import numpy as np

df = pd.read_csv('zoo.csv')
X = df.iloc[:,1:-1]
Y = df.iloc[:,-1]
oop = decision_tree()
print("The builded tree : ")
print(oop.fit(np.array(X).tolist(), np.array(Y).tolist(), 'e'))
print("Prediction for original data:")
print(oop.predict(np.array(X).tolist()))
print("Original labels:")
print(np.array(Y).tolist())
