from sklearn.ensemble import IsolationForest
import numpy as np
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
    data1 = data1[:100]
    data2 = data2[:1000]
    return np.array(data1), np.array(data2)
def calc_accres(y_true, y_pre):
    y_pre=[max(i,0) for i in y_pre]
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
    acc = TP / (TP + FP)
    rec = TP / (TP + FN)
    return acc, rec

def work(train_x,train_y,test_x,test_y):
    gbdt=IsolationForest()
    train_y=train_y.T
    train_y=train_y[0]
    gbdt.fit(train_x,train_y)
    y_pre=gbdt.predict(test_x)
    test_y=test_y.T
    test_y=test_y[0]
    return calc_accres(y_pre,test_y)

if __name__=="__main__":
    from sklearn import preprocessing

    # 读取数据
    train_data, test_data = read_data()
    m = len(train_data[0])
    min_max_scaler_train = preprocessing.MinMaxScaler()

    train_x = min_max_scaler_train.fit_transform(train_data[:, :-1])
    train_y = train_data[:, m - 1: m]

    test_x = min_max_scaler_train.fit_transform(test_data[:, :-1])
    test_y = test_data[:, m - 1: m]

    iso_acc, iso_rec = work(train_x, train_y, test_x, test_y)
    print(iso_acc,iso_rec)