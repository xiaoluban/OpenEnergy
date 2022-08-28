import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from datetime import datetime

# scaler对应5个，分别是order值，4个天气因素（温度，降雨，降雪，积雪）
def scaler_get(data):
    data_temp = np.copy(data)
    scaler = MinMaxScaler()#StandardScaler()#
    diff = data[1:, 1:2] - data[:data.shape[0] - 1, 1:2]

    diff[diff < 0] = -1
    diff[diff > 0] = 1

    diff = np.concatenate((np.zeros((1, 1)), diff), 0)
    data = np.concatenate((diff, data[:, 1:2]), 1)
    data = np.concatenate((data_temp[:, 0:1], data), 1)

    data = np.concatenate((data, data_temp[:, 2:]), 1)

    scaler.fit(data[:, 2:])
    data[:, 2:] = scaler.transform(data[:, 2:])
    return scaler


class customdataset(Dataset):
    def __init__(self, data, obs_day, pred_day, scaler, infer):
        self.infer = infer
        self.data = data
        if not self.infer:
            self.scaler = scaler#StandardScaler()
        else:
            self.scaler = scaler

        self.obs = obs_day
        self.pred = pred_day
        self.read_data()



    def read_data(self):
        # data = self.data
        day = self.obs + self.pred

        diff = self.data[1:, 1:2] - self.data[:self.data.shape[0]-1, 1:2]
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.plot(np.arange(0, diff.shape[0], 1), diff, marker='o', color='green', label=' diff')
        # ax1.set_ylabel('difference(green)')

        diff[diff<=0] = -0.1
        diff[diff>0] = 0.1

        # ax2 = ax1.twinx()
        # ax2.plot(np.arange(0, diff.shape[0], 1), diff, marker='*', color='red', label='modified diff')
        # ax2.set_ylabel('modified difference (red)')
        # #ax.set_xticks(np.arange(0, diff.shape[0], 1))
        # fig.legend()
        # plt.show()
        # data数据的列 ： 具体时间，difference， order数据， week， 天气
        diff = np.concatenate((np.zeros((1,1)), diff), 0)
        data = np.concatenate((diff, self.data[:, 1:2]), 1)
        data = np.concatenate((self.data[:, 0:1], data), 1)
        data = np.concatenate((data, diff), 1)

        # 考虑周一到周日得时间
        for j in range(data.shape[0]):
            s = str(data[j, 0])
            date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
            date = (datetime.date(date).weekday() +1)#/7.0
            data[j, 3:4] = date
            # t = 0
        # 每一列分别是：
        data = data.astype(np.float64)
        data = np.concatenate((data, self.data[:, 2:]), 1)
        # t = [",".join(item) for item in data[:, 0:1].astype(str)]
        #
        # t = str(data[:, 0])
        # date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
        #
        # t = datetime.date(str(data[0, 0])).weekday()
        if not self.infer:
            # self.scaler.fit(data[:, 1:].reshape(data[:, 1:].shape[0]*data[:, 1:].shape[1], 1))
            # data_temp = self.scaler.transform(data[:, 1:].reshape(data[:, 1:].shape[0]*data[:, 1:].shape[1], 1))
            # data[:, 1:] = data_temp.reshape(data[:, 1:].shape[0], data[:, 1:].shape[1])

            # self.scaler.fit(data[:, 2:])
            data[:, [2, 4, 5, 6, 7]] = self.scaler.transform(data[:, [2, 4, 5, 6, 7]])
            # data[:, 1:2] = self.scaler.transform(data[:, 1:2])
            self.out = []
            i = 0
            while True:
                self.out.append(data[int(i):int(i + day), :])
                i = i+1
                if i > self.data.shape[0]-day:
                    break
        else:
            # self.scaler.fit(data[:, 1:].reshape(data[:, 1:].shape[0]*data[:, 1:].shape[1], 1))
            # data_temp = self.scaler.transform(data[:, 1:].reshape(data[:, 1:].shape[0]*data[:, 1:].shape[1], 1))
            # data[:, 1:] = data_temp.reshape(data[:, 1:].shape[0], data[:, 1:].shape[1])

            # self.scaler.fit(data[:, 2:])

            data[:, [2, 4, 5, 6, 7]] = self.scaler.transform(data[:, [2, 4, 5, 6, 7]])
            # data[:, 1:2] = self.scaler.transform(data[:, 1:2])

            self.out = []
            i = 0
            while True:
                # t = self.data[int(i):int(i + 12), :]
                self.out.append(data[int(i):int(i + day), :])
                i = i+self.obs
                if i > self.data.shape[0]-day:
                    break


    def __len__(self):
        return len(self.out)
    def __getitem__(self, item):
        return self.out[item]
