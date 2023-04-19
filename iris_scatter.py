import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
iris_data = iris.data

# print("iris_data", iris_data)
# print("shape", iris_data.shape)

# 散布図を表示

st_data = iris_data[:50]
vc_data = iris_data[50:100]

plt.scatter(st_data[:, 0], st_data[:, 1], label="Setosa")
plt.scatter(vc_data[:, 0], vc_data[:, 1], label="Versicolor")
plt.legend()

plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.show()