import numpy as np
from numpy import ndarray

from typing import Callable, Dict, Tuple, List

np.set_printoptions(precision=4)
# GRAPHS_IMG_FILEPATH = "/Users/seth/development/01_deep-learning-from-scratch/images/02_fundamentals/graphs/"

from sklearn.datasets import *

# boston = sklearn.datasets.fetch_openml('boston', return_X_y=True)
boston = load_diabetes()

print(boston)
data = boston.data
target = boston.target
features = boston.feature_names

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)


from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)
print(preds.shape)
print(y_test.shape)
import matplotlib.pyplot as plt

# plt.xlim([0, 51])
# plt.ylim([0, 51])
plt.scatter(preds, y_test)
# plt.plot([0, 51], [0, 51]);
plt.show()
# plt.savefig(IMG_FILEPATH + "00_linear_real_pred_vs_actual.png");