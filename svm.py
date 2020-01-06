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
        return np.exp(- self.gamma * np.sum((x1[None] - x2[:, None]) ** 2, axis=2))

    def select_alpha(self):
        # ランダムに２値を選択
        if self.select_method == 'random':
            kkt_fill = False
            idx1, idx2 = randint(0, self.data_num, 2)

        # KKT条件を満たしていないものから選択
        if self.select_method == 'kkt':
            kkt_fill = True  # 全ての点が　KKT条件を満たしていれば１にする
            idx_list = np.random.permutation(np.arange(self.data_num))
            idx2 = idx_list[0]

            # 各点についてKKT条件を満たしていないものを探す
            for i in idx_list:
                if 0 < self.alpha[i] < self.C and self.y_train[i] * self.pred[i] != 1:
                    kkt_fill = False
                elif self.alpha[i] == 0 and self.y_train[i] * self.pred[i] < 1:
                    kkt_fill = False
                elif self.alpha[i] == self.C and self.y_train[i] * self.pred[i] > 1:
                    kkt_fill = False

                if not kkt_fill:
                    idx2 = i
                    break

            # E[idx1]とE[idx2]の差が大きいものを選択
            idx1 = np.argmin(self.E) if self.E[idx2] >= 0 else np.argmax(self.E)

        return idx1, idx2, kkt_fill

    def calculate_alpha(self, idx1, idx2):
        # 最大値と最上値を定める
        if self.y_train[idx1] == self.y_train[idx2]:
            low = max(0.0, self.alpha[idx1] + self.alpha[idx2] - self.C)
            high = min(self.C, self.alpha[idx1] + self.alpha[idx2])
        else:
            low = max(0.0, self.alpha[idx2] - self.alpha[idx1])
            high = min(self.C, self.C + self.alpha[idx2] - self.alpha[idx1])

        k = self.kernel[self.kernel_shape](self.x_train[[idx1, idx2]], self.x_train[[idx1, idx2]])
        eta = 2 * k[0, 1] - k[0, 0] - k[1, 1]

        # αを更新
        alpha2new = self.alpha[idx2] - self.y_train[idx2] * (self.E[idx1] - self.E[idx2]) / (eta - 0.001)
        self.alpha[idx2] = np.clip(alpha2new, low, high)
        self.alpha[idx1] = self.alpha[idx1] + self.y_train[idx1] * self.y_train[idx2] * (self.alpha[idx2] - alpha2new)

    def calculate_y(self):
        self.b = 0
        # サポートベクトルの計算
        if np.sum(self.alpha > 0) > 0:
            # bの計算
            sv_idx = self.alpha > 0
            if self.kernel_shape == 'linear':
                self.b = np.mean(self.y_train[sv_idx] - np.dot(self.kernel[self.kernel_shape](self.x_train[sv_idx], self.x_train[sv_idx]), self.alpha[sv_idx] * self.y_train[sv_idx]))
            else:
                self.b = np.mean(self.y_train[sv_idx] - np.diag(self.kernel[self.kernel_shape](self.x_train[sv_idx], self.x_train[sv_idx])) * self.alpha[sv_idx] * self.y_train[sv_idx])

        # y(予測値)の計算
        self.pred = np.dot(self.kernel[self.kernel_shape](self.x_train, self.x_train), self.alpha * self.y_train) + self.b

        # E(予測値とラベルの差)を計算
        self.E = self.pred - self.y_train

    def calculate_y_test(self, test_data):
        return np.dot(self.kernel[self.kernel_shape](self.x_train, test_data), self.alpha * self.y_train) + self.b

    '''
    def calculate_one_y(self, x, x_temp):
        y = self.b
        if self.kernel_shape == 'linear':
            y += np.dot(self.kernel[self.kernel_shape](x, x_temp), self.alpha * self.y_train)
        else:
            for i in range(self.data_num):
                y += self.kernel[self.kernel_shape](x[i], x_temp) * self.alpha[i] * self.y_train[i]
        return y
    '''

    def accuracy(self, y, t):
        return np.mean(y * t > 0)

    def run(self):
        print('calculating SVM')
        for i in tqdm(range(self.epoch)):

            # 更新するαを２つ選択
            idx1, idx2, kkt_fill = self.select_alpha()
            # print(f'idx1 = {idx1}, idx2 = {idx2}')

            # 全ての点がKTT条件を満たしていれば終了
            if kkt_fill:
                break

            # αを更新
            self.calculate_alpha(idx1, idx2)

            # 予測値yなどを更新
            self.calculate_y()

        acc = self.accuracy(self.pred, self.y_train)
        return acc, self.alpha, self.b, self.pred

    def test(self):
        pred_test = self.calculate_y_test(self.x_test)
        acc_test = self.accuracy(pred_test, self.y_test)
        return acc_test

    def plot(self):
        grid_list = np.arange(-3, 3, 0.1)  # plotの細かさの設定
        grid = []
        for xg in grid_list:
            for yg in grid_list:
                grid.append([xg, yg])

        # グリッド状に結果をプロット
        grid = np.array(grid)
        y_plot = self.calculate_y_test(grid)
        plt.scatter(grid[y_plot > 0][:, 0], grid[y_plot > 0][:, 1], c='cyan', linewidth=0, alpha=0.5, s=10)
        plt.scatter(grid[y_plot <= 0][:, 0], grid[y_plot <= 0][:, 1], c='salmon', linewidth=0, alpha=0.5, s=10)

        # 元のデータをプロット
        plt.scatter(self.x_train[self.y_train == 1][:, 0], self.x_train[self.y_train == 1][:, 1], c='blue')
        plt.scatter(self.x_train[self.y_train == -1][:, 0], self.x_train[self.y_train == -1][:, 1], c='red')
        plt.show()

    def plot_pred(self):
        plt.scatter(self.x_train[self.pred > 0][:, 0], self.x_train[self.pred > 0][:, 1], c='blue')
        plt.scatter(self.x_train[self.pred <= 0][:, 0], self.x_train[self.pred <= 0][:, 1], c='red')
        plt.show()
