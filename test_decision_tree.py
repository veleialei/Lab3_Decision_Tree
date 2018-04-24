from decision_tree import decision_tree
import pandas as pd
import numpy as np

df = pd.read_csv('animals.csv')
X = df.iloc[:,1:-1]
Y = df.iloc[:,-1]
oop = decision_tree()
print("The builded tree : ")
print(oop.fit(np.array(X).tolist(), np.array(Y).tolist(), 'e'))
print("Prediction :")
print(oop.predict(np.array(X).tolist()))
