import json
import numpy as np
import torch
from torch.autograd import Variable as V
import torch.utils.data as Data

class data_loader:
    def __init__(self,config_path = 'taxi_data.json'):
        print(config_path)
        self.config = json.load(open(config_path, "r"))

    def load_data(self):
        train = np.load(self.config['volume_train'])
        test = np.load(self.config['volume_test'])
        volume_train = train['volume']
        volume_test = test['volume']
        training_inflow = {}
        training_outflow = {}
        testing_inflow = {}
        testing_outflow = {}

        for i in range(volume_train.shape[0]):
            training_inflow[i] = {}
            training_outflow[i] = {}
            for row in range(volume_train.shape[1]):
                for column in range(volume_train.shape[2]):
                    ids = row * volume_train.shape[2] + column
                    in_value = volume_train[i][row][column][0]
                    out_value = volume_train[i][row][column][1]
                    training_inflow[i][ids] = in_value * 1.0 / self.config['volume_train_max'][0]
                    training_outflow[i][ids] = out_value * 1.0 / self.config['volume_train_max'][1]

        for i in range(volume_test.shape[0]):
            testing_inflow[i] = {}
            testing_outflow[i] = {}
            for row in range(volume_test.shape[1]):
                for column in range(volume_test.shape[2]):
                    ids = row * volume_test.shape[2] + column
                    in_value = volume_test[i][row][column][0]
                    out_value = volume_test[i][row][column][1]
                    testing_inflow[i][ids] = in_value * 1.0 / self.config['volume_train_max'][0]
                    testing_outflow[i][ids] = out_value * 1.0 / self.config['volume_train_max'][1]

        training_data = []
        label = []
        validation_data = []
        validation_label = []
        testing_data = []
        testing_label = []

        start_data = int(len(training_inflow) * 0.2)
        for i in range(start_data, len(training_inflow) - 1):
            training_data.append([])
            label.append([])
            for j in range(200):
                location = np.zeros(200)
                location[j] = 1
                temporal = np.zeros(48)
                temporal[i % 48] = 1
                location = location.tolist()
                temporal = temporal.tolist()
                feature = []
                feature.extend(location)
                feature.extend(temporal)
                station_label = [training_inflow[i + 1][j], training_outflow[i + 1][j]]

                # add time slots for trend; previous time slots
                for slot_num in range(self.config['trend_num']):
                    for k in range(self.config['local_context_len']):
                        index = i - slot_num - k
                        if index < 0:
                            index = len(training_inflow) + index
                        feature.append(training_inflow[index][j])
                    for k in range(self.config['local_context_len']):
                        index = i - slot_num - k
                        if index < 0:
                            index = len(training_inflow) + index
                        feature.append(training_outflow[index][j])

                # add time slot for daily period
                for slot_num in range(1, self.config['daily_period_num']):
                    for k in range(self.config['local_context_len']):
                        index = i + 1 - slot_num * 48 - k
                        if index < 0:
                            index = len(training_inflow) + index
                        feature.append(training_inflow[index][j])
                    for k in range(self.config['local_context_len']):
                        index = i + 1 - slot_num * 48 - k
                        if index < 0:
                            index = len(training_inflow) + index
                        feature.append(training_outflow[index][j])

                '''for slot_num in range(1, self.config['weekly_period_num']):
                    for k in range(self.config['local_context_len']):
                        index = i + 1 - slot_num * 48 * 7 - k
                        if index < 0:
                            index =  len(training_inflow) + index - 5 * 48
                            if index < 0:
                                index =  len(training_inflow) + index - 5 * 48

                        feature.append(training_inflow[index][j])
                    for k in range(self.config['local_context_len']):
                        index = i + 1 - slot_num * 48 * 7 - k
                        if index < 0:
                            index = len(training_inflow) + index - 5 * 48
                            if index < 0:
                                index = len(training_inflow) + index - 5 * 48
                        feature.append(training_outflow[index][j])'''


                training_data[-1].append(feature.copy())
                label[-1].append(station_label.copy())

        for i in range(0, start_data):
            validation_data.append([])
            validation_label.append([])
            for j in range(200):
                location = np.zeros(200)
                location[j] = 1
                temporal = np.zeros(48)
                temporal[i % 48] = 1
                location = location.tolist()
                temporal = temporal.tolist()
                feature = []
                feature.extend(location)
                feature.extend(temporal)
                station_label = [training_inflow[i + 1][j], training_outflow[i + 1][j]]

                # add time slots for trend; previous time slots
                for slot_num in range(self.config['trend_num']):
                    for k in range(self.config['local_context_len']):
                        index = i - slot_num - k
                        if index < 0:
                            index = len(training_inflow) + index
                        feature.append(training_inflow[index][j])
                    for k in range(self.config['local_context_len']):
                        index = i - slot_num - k
                        if index < 0:
                            index = len(training_inflow) + index
                        feature.append(training_outflow[index][j])

                # add time slot for daily period
                for slot_num in range(1, self.config['daily_period_num']):
                    for k in range(self.config['local_context_len']):
                        index = i + 1 - slot_num * 48 - k
                        if index < 0:
                            index = len(training_inflow) + index
                        feature.append(training_inflow[index][j])
                    for k in range(self.config['local_context_len']):
                        index = i + 1 - slot_num * 48 - k
                        if index < 0:
                            index = len(training_inflow) + index
                        feature.append(training_outflow[index][j])

                '''for slot_num in range(1, self.config['weekly_period_num']):
                    for k in range(self.config['local_context_len']):
                        index = i + 1 - slot_num * 48 * 7 - k
                        if index < 0:
                            index =  len(training_inflow) + index - 5 * 48 
                            if index < 0:
                                index =  len(training_inflow) + index - 5 * 48

                        feature.append(training_inflow[index][j])
                    for k in range(self.config['local_context_len']):
                        index = i + 1 - slot_num * 48 * 7 - k
                        if index < 0:
                            index = len(training_inflow) + index - 5 * 48
                            if index < 0:
                                index = len(training_inflow) + index - 5 * 48
                        feature.append(training_outflow[index][j])'''

                validation_data[-1].append(feature.copy())
                validation_label[-1].append(station_label.copy())

        for i in range(0, len(testing_inflow) - 1):
            testing_data.append([])
            testing_label.append([])
            for j in range(200):
                location = np.zeros(200)
                location[j] = 1
                temporal = np.zeros(48)
                temporal[i % 48] = 1
                location = location.tolist()
                temporal = temporal.tolist()
                feature = []
                feature.extend(location)
                feature.extend(temporal)
                station_label = [testing_inflow[i + 1][j], testing_outflow[i + 1][j]]
                for slot_num in range(self.config['trend_num']):
                    for k in range(self.config['local_context_len']):
                        index = i - slot_num - k
                        if index < 0:
                            index = len(training_inflow) + index
                            feature.append(training_inflow[index][j])
                        else:
                            feature.append(testing_inflow[index][j])
                    for k in range(self.config['local_context_len']):
                        index = i - slot_num - k
                        if index < 0:
                            index = len(training_inflow) + index
                            feature.append(training_outflow[index][j])
                        else:
                            feature.append(testing_outflow[index][j])

                for slot_num in range(1, self.config['daily_period_num']):
                    for k in range(self.config['local_context_len']):
                        index = i + 1 - slot_num * 48 - k
                        if index < 0:
                            index = len(training_inflow) + index
                            feature.append(training_inflow[index][j])
                        else:
                            feature.append(testing_inflow[index][j])
                    for k in range(self.config['local_context_len']):
                        index = i + 1 - slot_num * 48 - k
                        if index < 0:
                            index = len(training_inflow) + index
                            feature.append(training_outflow[index][j])
                        else:
                            feature.append(testing_outflow[index][j])

                '''for slot_num in range(1, self.config['weekly_period_num']):
                    for k in range(self.config['local_context_len']):
                        index = i + 1 - slot_num * 48 * 7 - k
                        if index < 0:
                            index =  len(training_inflow) + index
                            if index < 0:
                                index = len(training_inflow) + index - 5 * 48
                            feature.append(training_inflow[index][j])
                        else:
                            feature.append(testing_inflow[index][j])
                    for k in range(self.config['local_context_len']):
                        index = i + 1 - slot_num * 48 * 7 - k
                        if index < 0:
                            index =  len(training_inflow) + index
                            if index < 0:
                                index = len(training_inflow) + index - 5 * 48
                            feature.append(training_outflow[index][j])
                        else:
                            feature.append(testing_outflow[index][j])'''


                testing_data[-1].append(feature.copy())
                testing_label[-1].append(station_label.copy())



        x1 = V(torch.FloatTensor(training_data))
        y1 = V(torch.FloatTensor(label))
        torch_dataset_1 = Data.TensorDataset(x1, y1)
        BATCH_SIZE = 8
        train_loader = Data.DataLoader(dataset=torch_dataset_1,  # torch TensorDataset format
                                       batch_size=BATCH_SIZE,  # mini batch size
                                       shuffle=True,  # 要不要打乱数据 (打乱比较好)
                                       num_workers=2,  # 多线程来读数据
                                       )

        x2 = V(torch.FloatTensor(testing_data))
        y2 = V(torch.FloatTensor(testing_label))
        torch_dataset_2 = Data.TensorDataset(x2, y2)
        test_loader = Data.DataLoader(dataset=torch_dataset_2,  # torch TensorDataset format
                                      batch_size=BATCH_SIZE,  # mini batch size
                                      shuffle=True,  # 要不要打乱数据 (打乱比较好)
                                      num_workers=2,  # 多线程来读数据
                                      )

        x3 = V(torch.FloatTensor(validation_data))
        y3 = V(torch.FloatTensor(validation_label))
        torch_dataset_3 = Data.TensorDataset(x3, y3)
        val_loader = Data.DataLoader(dataset=torch_dataset_3,  # torch TensorDataset format
                                     batch_size=BATCH_SIZE,  # mini batch size
                                     shuffle=True,  # 要不要打乱数据 (打乱比较好)
                                     num_workers=2,  # 多线程来读数据
                                     )
        return train_loader, test_loader, val_loader














