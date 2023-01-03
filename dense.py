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


def sigmod_derivate(x):  # 创建一个 sigmod 导数
    return x * (1 - x)


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
    data1 = data1[:1000]
    data2 = data2[:10000]
    return np.array(data1), np.array(data2)


class network(object):

    def __init__(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no

        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n

        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)

        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)

        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    # 前向传播
    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    # 做一次反向传播，并更新误差
    def back_propagate(self, case, label, learn, correct):

        # 首先做一次前向的传播
        self.predict(case)

        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            # Ej=sigmod′(Oj)∗(Tj−Oj)=Oj(1−Oj)(Tj−Oj)
            output_deltas[o] = sigmod_derivate(self.output_cells[o]) * error

        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmod_derivate(self.hidden_cells[h]) * error

        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change

        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                if self.sparse[i][h] == 1:
                    change = hidden_deltas[h] * self.input_cells[i]
                    self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                    self.input_correction[i][h] = change

        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2

        return error

    def train(self, cases, labels, limit=1000, learn=0.01, correct=0.1):

        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)


def calc_accres(y_true, y_pre):
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pre[i] == 1:
            TP += 1
        elif y_true[i] == 1 and y_pre[i] == 0:
            FN += 1
        elif y_true[i] == 0 and y_pre[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pre[i] == 1:
            FP += 1
    print(TP, FN, FP, TN)
    acc = TP / (TP + FP)
    rec = TP / (TP + FN)
    return acc, rec


def work(train_x, train_y, test_x, test_y):
    # 集成神经网路个数为p
    net_list = []
    p = 1
    for i in range(0, p):
        net_list.append(network(m - 1, int(m / 2), 1))

    for i in range(0, p):
        net_list[i].train(train_x, train_y, 1, 0.5, 0.1)

    # 在测试集合上进行测试
    oup = []
    for i in range(0, len(test_x)):
        s = 0
        print(str(i) + "/" + str(len(test_data_)))
        for j in range(0, len(net_list)):
            s = s + net_list[j].predict(test_data_[i])[0]
        if s > p / 2:
            oup.append(1)
        else:
            oup.append(0)
    return calc_accres(oup, test_y)
