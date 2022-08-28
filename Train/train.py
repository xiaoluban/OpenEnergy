import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from read_data import *
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from PoissonRegress import ObjectiveWrapper
from scipy.optimize import minimize
from utils import *
from net import pred_net

import japanize_matplotlib

obs_day = 5
pred_day = 5
torch.set_default_dtype(torch.float64)
torch.manual_seed(4)
torch.random.manual_seed(4)
np.random.seed(4)


def train_val_dataset(dataset, val_split=0.10):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def get_scaler(flag_list, flag_, flag2_):
    total_order = []
    for flag in flag_:
        for flag2 in flag2_:
            for flag1 in flag_list:
                # 读取天气信息
                data_wea = pd.read_csv('../wea_days/' + '2021_' + str(flag1) + '.csv')
                data_wea = data_wea[['日', '気温', '降水量', '降雪', '積雪']]
                data_wea['降水量'][data_wea['降水量'].str.contains('--')] = 0
                data_wea['降雪'][data_wea['降雪'].str.contains('--')] = 0
                data_wea['積雪'][data_wea['積雪'].str.contains('--')] = 0
                data_wea['日'][data_wea['日'].astype(str).str.len() == 1] = str(0) + data_wea['日'][data_wea['日'].astype(str).str.len() == 1].astype(str)
                if len(str(flag1)) == 1:
                    fla_month = str(2021) + str(0) + str(flag1)
                else:
                    fla_month = str(2021)  + str(flag1)
                data_wea['日'] = fla_month + data_wea['日'].astype(str)
                data_wea.rename(columns={'日':'受注日'}, inplace=True)

                data_order = np.load('../filter_shop_order_all/' + '松戸_' + str(flag1) + '_' +  flag + '_' + flag2 + '.npy', allow_pickle=True)
                data_order = pd.DataFrame(data_order)
                data_order.columns = ["店名漢字", '売上部門名＿漢字',"受注日",  "数量"]
                data_append_wea = pd.merge(data_order, data_wea, on=['受注日'], how='inner')

                total_order.append(data_append_wea)

    total_order = pd.concat(total_order)
    total_order = total_order.to_numpy()
    total_order = total_order[:, 2:].astype(np.float)
    scaler = scaler_get(total_order)
    return scaler

def get_train_data(flag_list, scaler,  flag_, flag2_):
    total_order = []
    for flag in flag_:
        for flag2 in flag2_:
            for flag1 in flag_list:
                # 读取天气信息
                data_wea = pd.read_csv('../wea_days/' + '2021_' + str(flag1) + '.csv')
                data_wea = data_wea[['日', '気温', '降水量', '降雪', '積雪']]
                data_wea['降水量'][data_wea['降水量'].str.contains('--')] = 0
                data_wea['降雪'][data_wea['降雪'].str.contains('--')] = 0
                data_wea['積雪'][data_wea['積雪'].str.contains('--')] = 0
                data_wea['日'][data_wea['日'].astype(str).str.len() == 1] = str(0) + data_wea['日'][data_wea['日'].astype(str).str.len() == 1].astype(str)
                if len(str(flag1)) == 1:
                    fla_month = str(2021) + str(0) + str(flag1)
                else:
                    fla_month = str(2021)  + str(flag1)
                data_wea['日'] = fla_month + data_wea['日'].astype(str)
                data_wea.rename(columns={'日':'受注日'}, inplace=True)

                data_order = np.load('../filter_shop_order_all/' + '松戸_' + str(flag1) + '_' +  flag + '_' + flag2 + '.npy', allow_pickle=True)
                data_order = pd.DataFrame(data_order)
                data_order.columns = ["店名漢字", '売上部門名＿漢字',"受注日",  "数量"]

                data_append_wea = pd.merge(data_order, data_wea, on=['受注日'], how='inner')


                total_order.append(data_append_wea)
    # 读取天气信息
    # total_wea = []
    # for flag1 in flag_list:
    #     data_wea = pd.read_csv('../wea_days/' + '2021_' + str(flag1) + '.csv')
    #     data_wea = data_wea[['日', '気温', '降水量', '降雪', '積雪']]
    #     data_wea['降水量'][data_wea['降水量'].str.contains('--')]= 0
    #     data_wea['降雪'][data_wea['降雪'].str.contains('--')] = 0
    #     data_wea['積雪'][data_wea['積雪'].str.contains('--')] = 0
    #     total_wea.append()
    #
    #     t = 0


    total_order = pd.concat(total_order)
    total_order = total_order.to_numpy()
    total_order = total_order[:, 2:].astype(np.float64)


    dataset_all = customdataset(total_order, obs_day, pred_day, scaler, False)
    # scaler = dataset_all.scaler
    loader_train = DataLoader(dataset_all, batch_size=1, shuffle=True)
    # loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    return loader_train, scaler

def get_test_data(flag_list, scaler, flag_, flag2_):
    total_order = []
    for flag in flag_:
        for flag2 in flag2_:
            for flag1 in flag_list:
                # 读取天气信息
                data_wea = pd.read_csv('../wea_days/' + '2021_' + str(flag1) + '.csv')
                data_wea = data_wea[['日', '気温', '降水量', '降雪', '積雪']]
                data_wea['降水量'][data_wea['降水量'].str.contains('--')] = 0
                data_wea['降雪'][data_wea['降雪'].str.contains('--')] = 0
                data_wea['積雪'][data_wea['積雪'].str.contains('--')] = 0
                data_wea['日'][data_wea['日'].astype(str).str.len() == 1] = str(0) + data_wea['日'][data_wea['日'].astype(str).str.len() == 1].astype(str)
                if len(str(flag1)) == 1:
                    fla_month = str(2021) + str(0) + str(flag1)
                else:
                    fla_month = str(2021)  + str(flag1)
                data_wea['日'] = fla_month + data_wea['日'].astype(str)
                data_wea.rename(columns={'日':'受注日'}, inplace=True)

                data_order = np.load('../filter_shop_order_all/' + '松戸_' + str(flag1) + '_' + flag + '_' + flag2 + '.npy',
                                     allow_pickle=True)
                data_order = pd.DataFrame(data_order)
                data_order.columns = ["店名漢字", '売上部門名＿漢字', "受注日", "数量"]
                data_append_wea = pd.merge(data_order, data_wea, on=['受注日'], how='inner')
                total_order.append(data_append_wea)

    total_order = pd.concat(total_order)
    total_order = total_order.to_numpy()
    total_order = total_order[:, 2:].astype(np.float64)

    dataset_all = customdataset(total_order, obs_day, pred_day, scaler, True)
    scaler = dataset_all.scaler
    loader_train = DataLoader(dataset_all, batch_size=1, shuffle=False)

    return loader_train, scaler

# 对每个店铺的订单预测
def main():
    flag_list = [11, 12]  #月份 7-12
    flag_list_test = [8]
    # train_flag = ["食パン",  "サンロイヤル",  "ミニアンパン",  "シュークリーム"]
    # train_flag2 = ['イオン柏店',イオン浪江, 'イオン船橋','イト－ヨ－カド－西日暮里', 'ウエルシア北柏', ,イオン高萩店, イオン鎌ヶ谷店, ７－１１東日暮里,ロ－ソン柏の葉キャンパスシティ]
    train_flag = ["ミニアンパン"]
    train_flag2 = ['イオン高萩店']

    test_flag = ["ミニアンパン"]
    test_flag2 = ['イオン高萩店']

    # total_order = []
    # total_ship = []
    # for flag1 in flag_list:
    #     data_order = np.load('../shop_order/' + '松戸_' + str(flag1) + '_' +  flag + '_' + flag2 + '.npy', allow_pickle=True)
    #     data_order = pd.DataFrame(data_order)
    #     data_order.columns = ["店名漢字", '売上部門名＿漢字',"受注日",  "数量"]
    #     total_order.append(data_order)
    #
    # total_order = pd.concat(total_order)
    # total_order = total_order.to_numpy()
    # total_order = total_order[:, 2:4].astype(np.float)
    #
    #
    # dataset_all = customdataset(total_order, obs_day)
    # scaler = dataset_all.scaler
    # dataset_train, dataset_test = train_val_dataset(dataset_all)
    # loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    # loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    scaler = get_scaler(flag_list+flag_list_test, train_flag + test_flag, train_flag2 + test_flag2)
    loader_train, _ = get_train_data(flag_list, scaler, train_flag, train_flag2)
    loader_test, _ = get_test_data(flag_list_test, scaler, test_flag, test_flag2)


    net = pred_net(obs_day, pred_day)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # , weight_decay=0.05

    huberloss = nn.HuberLoss(reduction='mean', delta=1.0)
    klloss = nn.KLDivLoss(reduction="batchmean")

    for _ in range(40):

        for index, data_input in enumerate(loader_train):
            # 预测阈值
            x = data_input[:, 0:obs_day, 2:3]
            y = data_input[:, obs_day:(obs_day+pred_day), 2:3]
            output  = net(x)
            loss = cal_loss(y, output, huberloss, klloss)

            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    output = []
    y_test = []
    time_gt = []
    out_stock = []
    out_sale = []

    # data4thres = []
    # for index, data_input in enumerate(loader_test):
    #     data4thres.append(data_input[0, :, 2:4].T)
    # data4thres = np.hstack(data4thres)
    # thres = thres_net(torch.from_numpy(data4thres))

    with torch.no_grad():
        for index, data_input in enumerate(loader_test):
            x_test = data_input[:, 0:obs_day, 2:3]
            y_test1 = data_input[:, obs_day:(obs_day+pred_day), 2:3]
            output1= net(x_test)
            output.append(output1)
            y_test.append(y_test1)


            time_gt1 = data_input[:, obs_day:(obs_day+pred_day), 0].numpy().astype(np.float64)
            time_gt1 = time_gt1.reshape(time_gt1.shape[0] * time_gt1.shape[1], 1)
            time_gt1 = time_gt1.astype(int)
            time_gt.append(time_gt1)

    time_gt = np.vstack(time_gt)
    output = torch.vstack(output).numpy()
    y_test = torch.vstack(y_test).numpy()


    #gt = scaler.inverse_transform(y_test.reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2]))[:,1:2]
    #output = scaler.inverse_transform(np.concatenate((output, output), 2).reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2]))[:, 1:2]

    # gt = scaler.inverse_transform(y_test)
    # output = scaler.inverse_transform(output)
    y_test = np.concatenate((y_test, np.zeros((y_test.shape[0], y_test.shape[1], 4))), 2)
    gt = scaler.inverse_transform(y_test.reshape(y_test.shape[0] * y_test.shape[1], y_test.shape[2]))
    gt = gt.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2])
    output = np.concatenate((output, np.zeros((output.shape[0], output.shape[1], 4))), 2)
    output = scaler.inverse_transform(output.reshape(output.shape[0]*output.shape[1], output.shape[2]))
    output = output.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2])




    # gt = scaler.inverse_transform(y_test.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2]))
    # gt = gt.reshape(gt.shape[0], gt.shape[1], 1)
    # output = scaler.inverse_transform(output.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2]))
    # output = output.reshape(gt.shape[0], gt.shape[1], 1)

    # out_stock = scaler.inverse_transform(out_stock.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2]))
    # out_stock = out_stock.reshape(gt.shape[0], gt.shape[1], 1)
    # out_sale = scaler.inverse_transform(out_sale.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2]))
    # out_sale = out_sale.reshape(gt.shape[0], gt.shape[1], 1)


    mse, rmse, mape, r2, mae = metrics(output, gt)

    # plot
    output = output[:, :, 0].reshape(output.shape[0]*output.shape[1], 1)
    gt = gt[:, :, 0].reshape(gt.shape[0] * gt.shape[1], 1)


    x_gt = np.arange(0, time_gt.shape[0], 1)
    fig, ax = plt.subplots(figsize=(30, 10))
    plt.plot(x_gt, output, color='r', marker='*', label='prediction by RNN')
    plt.plot(x_gt, gt, color='g', marker='o', label='ground truth')
    # plt.plot(x_gt, out_stock, color='b', linestyle='dashed', label='stock')
    # plt.plot(x_gt, out_sale, color='orange', linestyle='dashed', label='sale')

    ax.set_xticks(x_gt)
    ax.set_xticklabels(time_gt, rotation=90, fontsize=13)
    plt.legend()
    plt.title('order prediction results of' + test_flag[0] + 'for' + test_flag2[0] + 'by RNN')
    plt.text(2,int(np.max(gt)-1), 'mae:' + str(mae) + ', mape' + "{:.2f}".format(mape) + ',r2:' + "{:.2f}".format(r2), fontsize=10)
    # plt.margins(0.5)
    plt.subplots_adjust(bottom=0.25)

    plt.savefig('order prediction results of' + test_flag[0] + 'for' + test_flag2[0] + 'by RNN'+ '.png')

    plt.show()


    t = 0


# 对工厂的出货和收到订单预测
# def main():
#     flag_list = [8, 11, 12]  #月份 7-12
#     flag = "生ケ－キ"# "生ケ－キ" #"菓子パン" # "生ケ－キ" "外注惣菜" "ソフト食パン"
#     total_order = []
#     total_ship = []
#     for flag1 in flag_list:
#         data_order = np.load('../matsudo_order/' + '松戸_' + str(flag1) + '_' + flag + '.npy', allow_pickle=True)
#         data_ship = np.load('../matsudo_ship/' + '松戸_' + str(flag1) + '_' + flag + '.npy', allow_pickle=True)
#
#         data_order = pd.DataFrame(data_order)
#         data_ship = pd.DataFrame(data_ship)
#         data_order.columns = ["工場名＿漢字", "売上部門名＿漢字", "受注日",  "数量"]
#         data_ship.columns = ["工場名＿漢字", "売上部門名＿漢字", "出荷日",  "数量"]
#
#         total_order.append(data_order)
#         total_ship.append(data_ship)
#
#     total_ship = pd.concat(total_ship)
#     total_order = pd.concat(total_order)
#
#     total_order = total_order.to_numpy()
#     total_ship = total_ship.to_numpy()
#
#     total_order = total_order[:, 2:4].astype(np.float)
#     total_ship = total_ship[:, 2:4].astype(np.float)
#
#     dataset_all = customdataset(total_order, obs_day)
#     scaler = dataset_all.scaler
#     dataset_train, dataset_test = train_val_dataset(dataset_all)
#     loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
#     loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False)
#
#     net = pred_net()
#     # optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=0.05)
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # , weight_decay=0.05
#
#     for _ in range(100):
#         for index, data_input in enumerate(loader_train):
#             x = data_input[:, 0:obs_day, 1:2]
#             y = data_input[:, obs_day:, 1:2]
#             output = net(x)
#             loss = cal_loss(y, output)
#
#             print(loss)
#
#             optimizer.zero_grad()
#             loss.backward()
#             # torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
#             optimizer.step()
#
#     output = []
#     y_test = []
#     time_gt = []
#     with torch.no_grad():
#         for index, data_input in enumerate(loader_test):
#             x_test = data_input[:, 0:obs_day, 1:2]
#             y_test1 = data_input[:, obs_day:, 1:2]
#             output1 = net(x_test)
#             output.append(output1[:,:,0])
#             y_test.append(y_test1[:,:,0])
#
#             time_gt1 = data_input[:, obs_day:, 0].numpy()
#             time_gt1 = time_gt1.reshape(time_gt1.shape[0] * time_gt1.shape[1], 1)
#             time_gt1 = time_gt1.astype(int)
#             time_gt.append(time_gt1)
#
#     time_gt = np.vstack(time_gt)
#     output = torch.vstack(output).numpy()
#     y_test = torch.vstack(y_test).numpy()
#     output = scaler.inverse_transform(output)
#     gt = scaler.inverse_transform(y_test)
#
#
#     mse, rmse, mape, r2, mae = metrics(output, gt)
#
#     # plot
#     output = output.reshape(output.shape[0]*output.shape[1], 1)
#     gt = gt.reshape(gt.shape[0] * gt.shape[1], 1)
#     x_gt = np.arange(0, time_gt.shape[0], 1)
#     fig, ax = plt.subplots(figsize=(30, 10))
#     plt.plot(x_gt, output, color='r', marker='*', label='prediction by RNN')
#     plt.plot(x_gt, gt, color='g', marker='o', label='ground truth')
#     ax.set_xticks(x_gt)
#     ax.set_xticklabels(time_gt, rotation=90, fontsize=13)
#     plt.legend()
#     plt.title('order prediction results of' + flag + 'by RNN')
#     plt.text(2,400000, 'mae:' + str(mae) + ', mape' + str(mape) + ',r2:' + str(r2), fontsize=20)
#     # plt.margins(0.5)
#     plt.subplots_adjust(bottom=0.25)
#
#     plt.savefig('order prediction results of' + flag + 'by RNN' + '.png')
#
#     plt.show()
#
#
#     t = 0

def train_val_dataset(dataset, val_split=0.30):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


if __name__ == '__main__':
    main()
