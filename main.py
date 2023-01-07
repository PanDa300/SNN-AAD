import math
import random
import numpy as np
from sklearn import preprocessing


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=1.0):  # 创造一个指定大小的矩阵
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):  # 创建一个 sigmod 函数
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivate(x):  # 创建一个 sigmod 导数
    return x * (1 - x)


def get_0_1_array(m, n, rate=0.2):
    array = np.ones(m * n).reshape(m, n)
    zeros_num = int(array.size * rate)  # 根据0的比率来得到 0的个数
    new_array = np.ones(array.size)  # 生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0  # 将一部分换为0
    np.random.shuffle(new_array)  # 将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)  # 重新定义矩阵的维度，与模板相同
    return re_array.tolist()


def read_data():
    fileHandler1 = open("train_data", "r")
    fileHandler2 = open("test_data", "r")
    listOfLines1 = fileHandler1.readlines()
    listOfLines2 = fileHandler2.readlines()
    data1 = []
    data2 = []

    for line in listOfLines1:
        newline = line.strip("\n").split(",")
        newline_ = [float(x) for x in newline]
        data1.append(newline_)

    for line in listOfLines2:
        newline = line.strip("\n").split(",")
        newline_ = [float(x) for x in newline]
        data2.append(newline_)
    data1 = data2[:100]
    data2 = data2[:1000]
    return np.array(data1), np.array(data2)


class network(object):
    def __init__(self, ni, nh, no):
        self.input_n = ni + 1  # 输入层+偏置项
        self.hidden_n = nh
        self.output_n = no
        self.input_cells = [1.0] * self.input_n
        self.hidden1_cells = [1.0] * self.hidden_n
        self.hidden2_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n

        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.hidden_weights = make_matrix(self.hidden_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        self.sparse_input = np.ones((self.input_n, self.hidden_n))
        self.sparse_hidden = np.ones((self.hidden_n, self.hidden_n))

        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)

        for i in range(self.hidden_n):
            for h in range(self.hidden_n):
                self.hidden_weights[i][h] = rand(-1, 1)

        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)

        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.hidden_correction = make_matrix(self.hidden_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def purn(self):
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                if abs(self.input_weights[i][h]) < 0.02:
                    self.input_weights[i][h] = 0
                    self.sparse_input[i][h] = 0
        for i in range(self.hidden_n):
            for h in range(self.hidden_n):
                if abs(self.hidden_weights[i][h]) < 0.1:
                    self.hidden_weights[i][h] = 0
                    self.sparse_hidden[i][h] = 0

    def predict(self, inputs):
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]

        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden1_cells[j] = sigmoid(total)

        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.hidden_n):
                total += self.hidden1_cells[i] * self.hidden_weights[i][j]
            self.hidden2_cells[j] = sigmoid(total)

        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden2_cells[j] * self.output_weights[j][k]  # + self.bias_output[k]

            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # 计算得到输出output_cells
        self.predict(case)
        output_deltas = [0.0] * self.output_n
        error = 0.0
        # 计算误差 = 期望输出-实际输出
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]  # 正确结果和预测结果的误差：0,1，-1
            output_deltas[o] = sigmoid_derivate(self.output_cells[o]) * error  # 误差稳定在0~1内

        hidden2_deltas = [0.0] * self.hidden_n
        for j in range(self.hidden_n):
            error = 0.0
            for k in range(self.output_n):
                error += output_deltas[k] * self.output_weights[j][k]
            hidden2_deltas[j] = sigmoid_derivate(self.hidden2_cells[j]) * error

        hidden1_deltas = [0.0] * self.hidden_n
        for j in range(self.hidden_n):
            error = 0.0
            for k in range(self.hidden_n):
                error += hidden2_deltas[k] * self.hidden_weights[j][k]
            hidden1_deltas[j] = sigmoid_derivate(self.hidden1_cells[j]) * error

        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden2_cells[h]
                # 调整权重：上一层每个节点的权重学习*变化+矫正率
                self.output_weights[h][o] += learn * change

        for i in range(self.hidden_n):
            for h in range(self.hidden_n):
                if self.sparse_hidden[i][h] == 1:
                    change = hidden2_deltas[h] * self.hidden1_cells[i]
                    self.hidden_weights[i][h] += learn * change

        for i in range(self.input_n):
            for h in range(self.hidden_n):
                if self.sparse_input[i][h] == 1:
                    change = hidden1_deltas[h] * self.input_cells[i]
                    self.input_weights[i][h] += learn * change

        error = 0
        for o in range(len(label)):
            for k in range(self.output_n):
                error += 0.5 * (label[o] - self.output_cells[k]) ** 2

        return error

    def train(self, cases, labels, limit, learn, correct=0.1):

        for i in range(limit):
            error = 0.0
            # learn = le.arn_speed_start /float(i+1)
            for j in range(len(cases)):
                case = cases[j]
                label = labels[j]

                error += self.back_propagate(case, label, learn, correct)
            print(self.hidden_weights[0][0], self.hidden_correction[0][0], self.hidden1_cells[0])
            print("error:", error)


def manual_label(net_list, test_x, test_y):
    dic = {}
    p = len(net_list)
    lis = []
    for i in range(0, len(test_x)):
        abnormal = 0
        normal = 0
        for j in range(0, p):
            if net_list[j].predict(test_x[i])[0] > 0.5:
                abnormal = abnormal + 1
            else:
                normal = normal + 1
        dic[i] = math.fabs(abnormal - normal)
        # 选出分歧最大的交给人工标注
        lis = sorted(dic.items(), key=lambda x: x[1])
    for k in range(0, 10):
        b = test_data[lis[k][0]][len(test_data[lis[k][0]]) - 1]
        b = int(b)
        manu_data = [test_x[lis[k][0]]]
        manu_label = [[b]]
        for i in range(0, p):
            net_list[i].train(manu_data, manu_label, 1, 0.1, 0.1)  # 再学习


def calc_accres(y_true, y_pre):
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(y_true)):
        print(y_true[i], y_pre[i])
        if y_true[i] == 0 and y_pre[i] == 0:
            TP += 1
        elif y_true[i] == 0 and y_pre[i] == 1:
            FN += 1
        elif y_true[i] == 1 and y_pre[i] == 1:
            TN += 1
        elif y_true[i] == 1 and y_pre[i] == 0:
            FP += 1
    print(TP, FN, FP, TN)
    acc = TP / (TP + FP)
    rec = TP / (TP + FN)
    return acc, rec


def work(train_x, train_y, test_x, test_y):
    # 集成神经网路个数为p
    net_list = []
    p = 5
    for i in range(0, p):
        net_list.append(network(m - 1, int(m / 2), 1))

    for i in range(0, p):
        net_list[i].train(train_x, train_y, 1, 0.1, 0.1)
    for i in range(0, p):
        net_list[i].purn()
    for i in range(0, p):
        net_list[i].train(train_x, train_y, 1, 0.1, 0.1)

    manual_label(net_list, train_x, train_y)

    # 在测试集合上进行测试
    oup = []
    for i in range(0, len(test_x)):
        s = 0
        print(str(i) + "/" + str(len(test_x)))
        for j in range(0, len(net_list)):
            s = s + net_list[j].predict(test_x[i])[0]
        if s > p / 2:
            oup.append(1)
        else:
            oup.append(0)
    return calc_accres(oup, test_y)


if __name__ == "__main__":
    # 读取数据
    train_data, test_data = read_data()
    m = len(train_data[0])
    min_max_scaler_train = preprocessing.MinMaxScaler()

    train_x = min_max_scaler_train.fit_transform(train_data[:, :-1])
    train_y = train_data[:, m - 1: m]

    test_x = min_max_scaler_train.fit_transform(test_data[:, :-1])
    test_y = test_data[:, m - 1: m]
    res = work(train_x, train_y, test_x, test_y)
    print(res)
