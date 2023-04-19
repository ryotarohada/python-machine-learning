import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
iris_data = iris.data
sl_data = iris_data[:100, 0] # length
sw_data = iris_data[:100, 1] # width

# 平均値を0に
sl_ave = np.average(sl_data) # 平均値
sl_data -= sl_ave # 平均値を引く
sw_ave = np.average(sw_data)
sw_data -= sw_ave

# 入力をリストに格納
input_data = []
for i in range(100):
    input_data.append([sl_data[i], sw_data[i]])

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Neuron:
    def __init__(self):
        self.input_sum = 0.0
        self.output = 0.0

    def set_input(self, inp):
        self.input_sum += inp

    def get_output(self):
        self.output = sigmoid(self.input_sum)
        return self.output
    
    def reset(self):
        self.input_sum = 0.0
        self.output = 0.0

class NeuralNetwork:
    def __init__(self):

        # 重み
        self.weight_im = [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]] # 入力層から中間層への重み 入力:2 ニューロン数:3
        self.weight_mo = [[1.0, -1.0, 1.0]] # 中間層から出力層への重み 入力:2 ニューロン数:1

        # バイアス
        self.bias_im = [3.0, 0.0, -3.0] # 入力層から中間層へのバイアス ニューロン数:3
        self.bias_mo = [-0.5] # 中間層から出力層へのバイアス ニューロン数:1

        # 各層の初期化
        self.input_layer = [0.0, 0.0]
        self.middle_layer = [Neuron(), Neuron(), Neuron()]
        self.output_layer = [Neuron()]


    def commit(self, input_data):
        
        # 各層のリセット
        self.input_layer[0] = input_data[0] # 入力層は値を受け取るのみ
        self.input_layer[1] = input_data[1]
        self.middle_layer[0].reset()
        self.middle_layer[1].reset()
        self.middle_layer[2].reset()
        self.output_layer[0].reset()

        # 入力層から中間層への信号伝達
        self.middle_layer[0].set_input(self.input_layer[0] * self.weight_im[0][0])
        self.middle_layer[0].set_input(self.input_layer[1] * self.weight_im[0][1])
        self.middle_layer[0].set_input(self.bias_im[0])

        self.middle_layer[1].set_input(self.input_layer[0] * self.weight_im[1][0])
        self.middle_layer[1].set_input(self.input_layer[1] * self.weight_im[1][1])
        self.middle_layer[1].set_input(self.bias_im[1])

        self.middle_layer[2].set_input(self.input_layer[0] * self.weight_im[2][0])
        self.middle_layer[2].set_input(self.input_layer[1] * self.weight_im[2][1])
        self.middle_layer[2].set_input(self.bias_im[2])

        # 中間層から出力層への信号伝達
        self.output_layer[0].set_input(self.middle_layer[0].get_output() * self.weight_mo[0][0])
        self.output_layer[0].set_input(self.middle_layer[1].get_output() * self.weight_mo[0][1])
        self.output_layer[0].set_input(self.middle_layer[2].get_output() * self.weight_mo[0][2])
        self.output_layer[0].set_input(self.bias_mo[0])
        
        return self.output_layer[0].get_output()

# ニューラルネットワークのインスタンス
neural_network = NeuralNetwork()

# 実行
st_predicted = [[], []] # Setosa
vc_predicted = [[], []] # Versicolor
for data in input_data:
    if neural_network.commit(data) < 0.5:
        st_predicted[0].append(data[0] + sl_ave)
        st_predicted[1].append(data[1] + sw_ave)
    else:
        vc_predicted[0].append(data[0] + sl_ave)
        vc_predicted[1].append(data[1] + sw_ave)

plt.scatter(st_predicted[0], st_predicted[1], label="Setosa")
plt.scatter(vc_predicted[0], vc_predicted[1], label="Versicolor")
plt.legend()

plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.title("Predicted")
plt.show()
