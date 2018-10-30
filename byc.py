import pandas as pd 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
data = pd.read_csv("risk_factors_cervical_cancer.csv")
X = data[data.columns[0:5]]
Y = data[data.columns[34]]
clf.fit(X, Y)
print(clf.score(X, Y))
