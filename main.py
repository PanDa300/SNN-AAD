import numpy as np
import math
import os
import pandas as pd
import random
import net
import iso
import dense
import GBDT
from sklearn.metrics import accuracy_score, recall_score, f1_score

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
    data1 = data1[:10]
    data2 = data2[:100]
    return np.array(data1), np.array(data2)


def calc(y_true, y_pre):
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


def test():
    from sklearn import preprocessing
    # 读取数据
    train_data, test_data = read_data()
    m = len(train_data[0])
    min_max_scaler_train = preprocessing.MinMaxScaler()

    train_x = min_max_scaler_train.fit_transform(train_data[:, :-1])
    train_y = train_data[:, m - 1: m]

    test_x = min_max_scaler_train.fit_transform(test_data[:, :-1])
    test_y = test_data[:, m - 1: m]

    iso_acc, iso_rec = iso.work(train_x, train_y, test_x, test_y)
    GBDT_acc, GBDT_rec = iso.work(train_x, train_y, test_x, test_y)
    dense_acc, dense_rec = iso.work(train_x, train_y, test_x, test_y)
    net_acc, net_rec = iso.work(train_x, train_y, test_x, test_y)
    return iso_acc, iso_rec, GBDT_acc, GBDT_rec, dense_acc, dense_rec, net_acc, net_rec


def draw():
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    x=[2,4,6,8,10]
    acc1=[0.43,0.56,0.65,0.76,0.7]
    acc2=[0.42,0.57,0.69,0.79,0.82]
    rec1=[0.45,0.58,0.67,0.77,0.74]
    rec2=[0.44,0.55,0.67,0.75,0.78]
    plt.rcParams['font.family'] = 'STSong'  # 修改了全局变量
    ax1 = plt.subplot(121)
    ax1.scatter(x, acc1)
    ax1.plot(x, acc1, linewidth=3.0, label='Dense net')
    ax1.scatter(x, acc2)
    ax1.plot(x, acc2, linewidth=3.0, label='AADSNN')
    ax1.set_xlabel("迭代次数", fontsize=25)
    ax1.set_ylabel("准确率", fontsize=25)
    ax1.set_xticks(x)
    y = np.arange(0.2, 0.95, 0.1)
    ax1.set_yticks(y)
    ax1.tick_params(labelsize=17)  # 刻度字体大小13
    ax1.legend(prop={'size': 20})

    ax2 = plt.subplot(122)
    ax2.scatter(x, rec1)
    ax2.plot(x, rec1, linewidth=3.0, label='Dense net')
    ax2.scatter(x, rec2)
    ax2.plot(x, rec2, linewidth=3.0, label='AADSNN')
    ax2.set_xlabel("迭代次数", fontsize=25)
    ax2.set_ylabel("召回率", fontsize=25)
    ax2.set_xticks(x)
    y = np.arange(0.2, 0.95, 0.1)
    ax2.set_yticks(y)
    ax2.tick_params(labelsize=17)  # 刻度字体大小13
    ax2.legend(prop={'size': 20})

    plt.show()


if __name__ == "__main__":
    acc1 = []
    acc2 = []
    rec1 = []
    rec2 = []
    draw()
