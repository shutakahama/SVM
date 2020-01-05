import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *
from tqdm import tqdm


class SVM:
    def __init__(self, x_train, y_train, x_test, y_test, args):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.select_method = args.select_method
        self.C = args.C
        self.data_num = args.data_num
        self.gamma = args.gamma
        self.epoch = args.epoch

        self.alpha = np.zeros(self.data_num)
        self.E = - self.y_train
        self.pred = np.zeros(self.data_num)
        self.b = 0

        # kernelの種類の選択
        self.kernel_shape = args.kernel_shape
        self.kernel = {
            'linear': self.linear_kernel,
            'gaussian': self.gaussian_kernel
        }

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)

    def gaussian_kernel(self, x1, x2):
        diff = x1 - x2
        return np.exp(- self.gamma * np.dot(diff, diff.T))

    def select_alpha(self):
        # ランダムに２値を選択
        if self.select_method == 'random':
            kkt_fill = 0
            idx1, idx2 = randint(0, self.data_num, 2)

        # KKT条件を満たしていないものから選択
        if self.select_method == 'kkt':
            ktt_fill = 1  # 全ての点が　KKT条件を満たしていれば１にする
            idx_list = np.random.permutation(np.arange(self.data_num))
            idx2 = idx_list[0]

            # 各点についてKKT条件を満たしていないものを探す
            for idx_pre in idx_list:
                if 0 < self.alpha[idx_pre] < self.C and self.y_train[idx_pre] * self.pred[idx_pre] != 1:
                    kkt_fill = 0
                    idx2 = idx_pre
                    break
                elif self.alpha[idx_pre] == 0 and self.y_train[idx_pre] * self.pred[idx_pre] < 1:
                    kkt_fill = 0
                    idx2 = idx_pre
                    break
                elif self.alpha[idx_pre] == self.C and self.y_train[idx_pre] * self.pred[idx_pre] > 1:
                    kkt_fill = 0
                    idx2 = idx_pre
                    break

            # E[idx1]とE[idx2]の差が大きいものを選択
            if self.E[idx2] >= 0:
                idx1 = np.argmin(self.E)
            else:
                idx1 = np.argmax(self.E)

        return idx1, idx2, kkt_fill

    def culculate_alpha(self, idx1, idx2):
        # 最大値と最上値を定める
        if self.y_train[idx1] == self.y_train[idx2]:
            low = max(0.0, self.alpha[idx1] + self.alpha[idx2] - self.C)
            high = min(self.C, self.alpha[idx1] + self.alpha[idx2])
        else:
            low = max(0.0, self.alpha[idx2] - self.alpha[idx1])
            high = min(self.C, self.C + self.alpha[idx2] - self.alpha[idx1])

        k11 = self.kernel[self.kernel_shape](self.x_train[idx1], self.x_train[idx1])
        k12 = self.kernel[self.kernel_shape](self.x_train[idx1], self.x_train[idx2])
        k22 = self.kernel[self.kernel_shape](self.x_train[idx2], self.x_train[idx2])
        eta = 2 * k12 - k11 - k22

        if eta == 0:
            eta = -0.001

        # α２を更新
        alpha2old = self.alpha[idx2]
        alpha2new = self.alpha[idx2] - self.y_train[idx2] * (self.E[idx1] - self.E[idx2]) / eta
        if alpha2new >= high:
            alpha2new = high
        elif alpha2new <= low:
            alpha2new = low

        # α１を更新
        alpha1new = self.alpha[idx1] + self.y_train[idx1] * self.y_train[idx2] * (alpha2old - alpha2new)
        self.alpha[idx1] = alpha1new
        self.alpha[idx2] = alpha2new

    def culculate_y(self):
        if np.sum(self.alpha > 0) == 0:
            self.b = 0
        else:
            self.b = 0
            # サポートベクトルのインデックス
            sv_idx = self.alpha > 0
            # bの計算
            if self.kernel_shape == 'linear':
                self.b = np.mean(self.y_train[sv_idx] - np.dot(self.kernel[self.kernel_shape](self.x_train[sv_idx], self.x_train[sv_idx]), self.alpha[sv_idx] * self.y_train[sv_idx]))
            else:
                x_sv = self.x_train[sv_idx]
                t_sv = self.y_train[sv_idx]
                alpha_sv = self.alpha[sv_idx]
                for i in range(len(x_sv)):
                    self.b += t_sv[i] - self.kernel[self.kernel_shape](x_sv[i], x_sv[i]) * alpha_sv[i] * t_sv[i]
                self.b /= len(x_sv)

        # y(予測値)の計算
        if self.kernel_shape == 'linear':
            self.pred = np.dot(self.kernel[self.kernel_shape](self.x_train, self.x_train), self.alpha * self.y_train) + np.full(self.data_num, b)
        else:
            self.pred = np.full(self.data_num, self.b)
            for j in range(self.data_num):
                for k in range(self.data_num):
                    self.pred[j] += self.kernel[self.kernel_shape](self.x_train[j], self.x_train[k]) * self.alpha[k] * self.y_train[k]

        # E(予測値とラベルの差)を計算
        self.E = self.pred - self.y_train

    def culculate_y_test(self):
        pred_test = np.full(self.data_num, self.b)
        for j in range(self.data_num):
            for k in range(self.data_num):
                pred_test[j] += self.kernel[self.kernel_shape](self.x_train[k], self.x_test[j]) * self.alpha[k] * self.y_train[k]
        return pred_test

    def culculate_one_y(self, x, x_temp):
        y = self.b
        if self.kernel_shape == 'linear':
            y += np.dot(self.kernel[self.kernel_shape](x, x_temp), self.alpha * self.y_train)
        else:
            for i in range(self.data_num):
                y += self.kernel[self.kernel_shape](x[i], x_temp) * self.alpha[i] * self.y_train[i]
        return y

    def accuracy(self, y, t):
        return np.mean(y * t > 0)

    def run(self):
        print('calculating SVM')
        for i in range(self.epoch):
            print('\rroop {}'.format(i), end=' ')

            # 更新するαを２つ選択
            idx1, idx2, kkt_fill = self.select_alpha()
            # print('idx1 = {}, idx2 = {}'.format(idx1, idx2))

            # 全ての点がKTT条件を満たしていれば終了
            if kkt_fill == 1:
                break

            # αを更新
            self.culculate_alpha(idx1, idx2)
            # print('alpha={}'.format(alpha[:100]))

            # 予測値yなどを更新
            self.culculate_y()

        # print('alpha={}'.format(alpha[:100]))
        print('\n')
        acc = self.accuracy(self.pred, self.y_train)
        return acc, self.alpha, self.b, self.pred

    def test(self):
        pred_test = self.culculate_y_test()
        acc_test = self.accuracy(pred_test, self.y_test)
        return acc_test

    def plot(self):
        x_grid_list = np.arange(-3, 3, 0.1)  # plotの細かさの設定
        y_grid_list = np.arange(-3, 3, 0.1)

        for xg in tqdm(x_grid_list):  # グラフ内の領域の点全てに関してplot
            for yg in y_grid_list:
                x_temp = np.array([xg, yg])
                y_plot = self.culculate_one_y(self.x_train[:, :2], x_temp)
                if y_plot > 0:
                    plt.scatter(xg, yg, c='cyan', linewidth=0, alpha=0.5, s=10)
                else:
                    plt.scatter(xg, yg, c='salmon', linewidth=0, alpha=0.5, s=10)

        for dot in range(self.data_num):  # 正解データのplot
            if self.y_train[dot] == 1:
                plt.scatter(self.x_train[dot][0], self.x_train[dot][1], c='blue')
            else:
                plt.scatter(self.x_train[dot][0], self.x_train[dot][1], c='red')
        plt.show()

    def plot_pred(self):
        for dot in range(self.data_num):
            if self.pred[dot] > 0:
                plt.scatter(self.x_train[dot][0], self.x_train[dot][1], c='blue')
            else:
                plt.scatter(self.x_train[dot][0], self.x_train[dot][1], c='red')
        plt.show()
