import time
import argparse

import dataset
from svm import SVM


'''
def boosting(x,t,sample_num=20): #sample_num : 各識別器で使うデータ数
    index_1 = np.random.permutation(np.arange(N)[:sample_num])
    x_1 = x[index_1]
    t_1 = t[index_1]
    print('fase1')
    acc_1,alpha_1,b_1,pred_1 = SVM(x_1,t_1) #第一の識別器の生成
    #print(acc_1)
    #plot(x_1,t_1,alpha_1,b_1)
    #plot_pred(x_1,pred_1)
    index_2pre = np.random.permutation(np.arange(N))
    index_2 = np.zeros(0)
    for num2 in index_2pre: #第二の識別器のデータを選択
        y_cand = culculate_one_y(x_1,x[num2],alpha_1,t_1,b_1)
        rd = randn()
        if rd>0.5 and y_cand*t[num2]>0:
            index_2 = np.append(index_2,num2)
        elif rd<0.5 and y_cand*t[num2]<0:
            index_2 = np.append(index_2,num2)
        if len(index_2) >= sample_num:
            break
    index_2 = index_2.astype(np.int8)
    x_2 = x[index_2]
    t_2 = t[index_2]
    print('fase2')
    acc_2,alpha_2,b_2,pred_2 = SVM(x_2,t_2) #第二の識別器の生成
    #print(acc_2)
    #plot(x_2,t_2,alpha_2,b_2)
    #plot_pred(x_2,pred_2)
    index_3pre = np.random.permutation(np.arange(N))
    index_3 = np.zeros(0)
    for num3 in index_3pre: #第三の識別器のデータを選択
        y_cand1 = culculate_one_y(x_1,x[num3],alpha_1,t_1,b_1)
        y_cand2 = culculate_one_y(x_2,x[num3],alpha_2,t_2,b_2)
        if y_cand1*y_cand2<0:
            index_3 = np.append(index_3,num3)
        if len(index_3) >= sample_num:
            break
    index_3 = index_3.astype(np.int8)
    x_3 = x[index_3]
    t_3 = t[index_3]
    print('fase3')
    acc_3,alpha_3,b_3,pred_3 = SVM(x_3,t_3) #第三の識別器の生成
    #print(acc_3)
    #plot(x_3,t_3,alpha_3,b_3)
    #plot_pred(x_3,pred_3)
    pred = np.zeros(N)
    for numL in range(N): #最終的な識別(３つの多数決)
        y_cand1 = culculate_one_y(x_1,x[numL],alpha_1,t_1,b_1)
        y_cand2 = culculate_one_y(x_2,x[numL],alpha_2,t_2,b_2)
        y_cand3 = culculate_one_y(x_3,x[numL],alpha_3,t_3,b_3)
        #print([y_cand1,y_cand2,y_cand3])
        if np.sum(np.array([y_cand1,y_cand2,y_cand3]) > 0) >= 2:
            pred[numL] = 1
        else:
            pred[numL] = -1
    #print(pred)

    return accuracy(pred,t),pred

def adaboost(x,t,sample_num=50):
    weight = np.ones(N)/N*sample_num
    print('sample sum is expected to be {}'.format(sample_num))
    m = 10
    y_L = np.zeros(0)
    alp_L = np.zeros(0)
    for times in range(m):
        print('fase {}'.format(times+1))
        sample_list = np.zeros(0)
        for idx in range(N): #各サンプルの重み付きサンプリング
            if rand()<weight[idx]:
                sample_list= np.append(sample_list,idx)
        print('sample_list_len {}'.format(len(sample_list)))
        sample_list = sample_list.astype(np.int8)
        x_tmp = x[sample_list]
        t_tmp = t[sample_list]
        acc_tmp,alpha_tmp,b_tmp,pred_tmp = SVM(x_tmp,t_tmp) #識別器の生成
        #plot_pred(x_tmp,pred_tmp)
        y_tmp = np.zeros(N)
        for num in range(N): # 全データの識別
            y_tmp[num] = culculate_one_y(x_tmp,x[num],alpha_tmp,t_tmp,b_tmp)
        I_tmp = y_tmp*t<0 #予測が間違っているもののインデックス
        eps = np.dot(weight,I_tmp)/np.sum(weight)
        alp = np.log((1-eps)/eps)/2.0 #識別器の重み
        #print('eps {}'.format(eps))
        #print('alp {}'.format(alp))
        weight = weight*np.exp(-alp*y_tmp*t) #各データの重みの更新
        weight = weight*sample_num/np.sum(weight)
        #print('weight= {}'.format(weight))
        y_L = np.append(y_L,y_tmp)
        alp_L = np.append(alp_L,alp)
    y_L = y_L.reshape(-1,100)
    #print('weight= {}'.format(weight))
    #print('alpha= {}'.format(alp_L))
    pred = np.dot(alp_L,y_L)

    return accuracy(pred,t),pred
'''

'''
acc_list = np.zeros(30)
tm_list = np.zeros(len(acc_list))
acc_test_list = np.zeros(len(acc_list))
for i in range(len(acc_list)):
    print('roop {}'.format(i))
    acc_list[i],acc_test_list[i],tm_list[i] = main(x_data,y_data,x_test,y_test,play='svm')

print('acc_train_ave = {}'.format(np.mean(acc_list)))
print('acc_test_ave = {}'.format(np.mean(acc_test_list)))
print('time_ave = {}'.format(np.mean(tm_list)))
'''


def main():
    parser = argparse.ArgumentParser(description='svm')
    parser.add_argument('--dataset', type=str, default='normal')  # データセットのタイプ
    parser.add_argument('--data_num', type=int, default=100)  # データ数
    parser.add_argument('--play', type=str, default='svm')  # 行う操作
    parser.add_argument('--C', type=int, default=0.1)  #
    parser.add_argument('--epoch', type=int, default=50)  #
    parser.add_argument('--gamma', type=int, default=10)  #
    parser.add_argument('--select_method', type=str, default='kkt')  #
    parser.add_argument('--kernel_shape', type=str, default='gaussian')  #
    args = parser.parse_args()

    # データ読み込み
    x_train, y_train, x_test, y_test = dataset.generate_data(dataset=args.dataset, data_num=args.data_num)
    # dataset.data_plot(x_train, y_train)

    start = time.time()
    if args.play == 'svm':
        model = SVM(x_train, y_train, x_test, y_test, args)
        acc, alpha, b, pred = model.run()
    # elif args.play == 'boosting':
    #     acc, pred = boosting(x_train, y_train)
    # elif args.play == 'adaboost':
    #     acc, pred = adaboost(x_train, y_train)
    tm = time.time() - start

    print(f'time {tm} [sec]')
    print(f'train accuracy={acc}')
    if args.play == 'svm':
        acc_test = model.test()  # テストデータで性能を検証
        print(f'test accuracy={acc_test}')
        model.plot()

    model.plot_pred()

    return acc, acc_test, tm


if __name__ == '__main__':
    main()

