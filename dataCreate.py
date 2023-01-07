import random


# covtype.data
from sklearn import preprocessing



def harData_0_1():
    train_data = []
    test_data = []

    fileHandler = open("har.data", "r")
    listOfLines = fileHandler.readlines()

    fp1 = open("train_data", 'w')
    fp2 = open("test_data", 'w')

    # 正常点
    c = 0
    for lines in listOfLines:
        line = lines.strip("\n").split(",")
        if int(line[561]) <= 4:
            if c < 645:
                line[561] = '0'
                train_data.append(line)
            if 645 <= c < 6449:
                line[561] = '0'
                test_data.append(line)
            c = c + 1

    c = 0
    for lines in listOfLines:
        line = lines.strip("\n").split(",")
        if line[561] == '6':
            if c < 32:
                line[561] = '1'
                train_data.append(line)
            if 32 <= c < 322:
                line[561] = '1'
                test_data.append(line)
            c = c + 1

    random.shuffle(train_data)
    random.shuffle(test_data)

    for i in range(0, len(train_data)):
        line = ""
        for j in range(0, len(train_data[i])):
            line = line + train_data[i][j] + ","
        fp1.write(line[:-1] + "\n")

    for i in range(0, len(test_data)):
        line = ""
        for j in range(0, len(test_data[i])):
            line = line + test_data[i][j] + ","
        fp2.write(line[:-1] + "\n")

def harData_0_0_5():
    train_data = []
    test_data = []

    fileHandler = open("har.data", "r")
    listOfLines = fileHandler.readlines()

    fp1 = open("train_data", 'w')
    fp2 = open("test_data", 'w')

    # 正常点
    c = 0
    for lines in listOfLines:
        line = lines.strip("\n").split(",")
        if int(line[561]) <= 4:
            if c < 323:
                line[561] = '0'
                train_data.append(line)
            if 323 <= c < 6449:
                line[561] = '0'
                test_data.append(line)
            c = c + 1

    c = 0
    for lines in listOfLines:
        line = lines.strip("\n").split(",")
        if line[561] == '6':
            if c < 16:
                line[561] = '1'
                train_data.append(line)
            if 16 <= c < 322:
                line[561] = '1'
                test_data.append(line)
            c = c + 1

    random.shuffle(train_data)
    random.shuffle(test_data)

    for i in range(0, len(train_data)):
        line = ""
        for j in range(0, len(train_data[i])):
            line = line + train_data[i][j] + ","
        fp1.write(line[:-1] + "\n")

    for i in range(0, len(test_data)):
        line = ""
        for j in range(0, len(test_data[i])):
            line = line + test_data[i][j] + ","
        fp2.write(line[:-1] + "\n")


def covtypeData_0_1():
    train_data = []
    test_data = []

    fileHandler = open("covtype.data", "r")
    listOfLines = fileHandler.readlines()

    fp1 = open("train_data", 'w')
    fp2 = open("test_data", 'w')

    c = 0
    for lines in listOfLines:
        line = lines.strip("\n").split(",")
        if line[54] == '1':
            if c < 21197:
                line[54] = '0'
                train_data.append(line)
            if 21197 <= c < 211840:
                line[54] = '0'
                test_data.append(line)
            c = c + 1

    c = 0
    for lines in listOfLines:
        line = lines.strip("\n").split(",")
        if line[54] == '2':
            if c < 2096:
                line[54] = '1'
                train_data.append(line)
            if 2096 <= c < 21093:
                line[54] = '1'
                test_data.append(line)
            c = c + 1

    random.shuffle(train_data)
    random.shuffle(test_data)

    for i in range(0, len(train_data)):
        line = ""
        for j in range(0, len(train_data[i])):
            line = line + train_data[i][j] + ","
        fp1.write(line[:-1] + "\n")

    for i in range(0, len(test_data)):
        line = ""
        for j in range(0, len(test_data[i])):
            line = line + test_data[i][j] + ","
        fp2.write(line[:-1] + "\n")


def covtypeData_0_0_5():
    train_data = []
    test_data = []

    fileHandler = open("covtype.data", "r")
    listOfLines = fileHandler.readlines()

    fp1 = open("train_data", 'w')
    fp2 = open("test_data", 'w')

    c = 0
    for lines in listOfLines:
        line = lines.strip("\n").split(",")
        if line[54] == '1':
            if c < 10599:
                line[54] = '0'
                train_data.append(line)
            if 10599 <= c < 211840:
                line[54] = '0'
                test_data.append(line)
            c = c + 1

    c = 0
    for lines in listOfLines:
        line = lines.strip("\n").split(",")
        if line[54] == '2':
            if c < 1048:
                line[54] = '1'
                train_data.append(line)
            if 2096 <= c < 21093:
                line[54] = '1'
                test_data.append(line)
            c = c + 1

    random.shuffle(train_data)
    random.shuffle(test_data)

    for i in range(0, len(train_data)):
        line = ""
        for j in range(0, len(train_data[i])):
            line = line + train_data[i][j] + ","
        fp1.write(line[:-1] + "\n")

    for i in range(0, len(test_data)):
        line = ""
        for j in range(0, len(test_data[i])):
            line = line + test_data[i][j] + ","
        fp2.write(line[:-1] + "\n")

harData_0_0_5()