import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *


def generate_data(dataset, data_num):
    if dataset == "mnist":
        # MNISTデータの読み込み
        data_pre = np.loadtxt('data/train-images5000.txt').astype(np.float32) / 255.0
        label_pre = np.loadtxt('data/train-labels5000.txt').astype(np.int32)
        # print(data_pre.shape)

        # 数字を二つ選ぶ
        num1, num2 = 2, 7

        # 選択したラベル2つについて取り出す
        data1 = data_pre[label_pre == num1]
        data2 = data_pre[label_pre == num2]
        data = np.concatenate((data1, data2), axis=0)
        label = np.full(data1.shape[0], 1) + np.full(data1.shape[0], -1)

        # データ数を絞る
        perm = permutation(len(label))
        x_train = data[perm[:data_num]]
        y_train = label[perm[:data_num]]
        x_test = data[perm[-data_num:]]
        y_test = label[perm[-data_num:]]

    if dataset == "separate":
        # 二分されたデータ分布を生成
        x_train = randn(data_num,2)
        y_train = np.array([1 if x1 > x2 else - 1 for x1, x2 in x_train])

        # 境界線付近のデータを一部分離不能にする
        rate = 0.1
        for i in range(data_num):
            if x_train[i][0] > x_train[i][1] and x_train[i][0] - x_train[i][1] < 0.5:
                if randn() < rate:
                    y_train[i] *= -1
            if x_train[i][1] > x_train[i][0] and x_train[i][1] - x_train[i][0] < 0.5:
                if randn() < rate:
                    y_train[i] *= -1

        # テストデータ生成
        x_test = randn(data_num, 2)
        y_test = np.array([1 if x1 > x2 else - 1 for x1, x2 in x_test])

        for i in range(data_num):
            if x_test[i][0] > x_test[i][1] and x_test[i][0] - x_test[i][1] < 0.5:
                if randn() > rate:
                    y_test[i] *= -1
            if x_test[i][1] > x_test[i][0] and x_test[i][1] - x_test[i][0] < 0.5:
                if randn() > rate:
                    y_test[i] *= -1

    if dataset == "normal":
        # 円形のデータ分布を生成
        x_train = randn(data_num, 2)
        y_train = np.array([1 if np.linalg.norm(x) < 1 else -1 for x in x_train])

        # テストデータ生成
        x_test = randn(data_num, 2)
        y_test = np.array([1 if np.linalg.norm(x) < 1 else -1 for x in x_test])

    return x_train, y_train, x_test, y_test


# データプロット
def data_plot(data, label):
    plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], c='blue')
    plt.scatter(data[label == -1][:, 0], data[label == -1][:, 1], c='red')
    plt.show()
