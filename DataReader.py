import json
import os
import numpy as np

class DataReader:
    def __init__(self):
        pass

    def read_data_set(self):
        current_working_directory = os.path.dirname(os.path.realpath(__file__))
        json_file_path = current_working_directory + "/shipsnet.json"

        json_file = open(json_file_path)
        dataset = json.load(json_file)
        json_file.close()

        return dataset

    def get_data_from_data_set(self, dataset):
        data = np.array(dataset['data']).astype('uint8')
        labels =  np.array(dataset['labels']).reshape(len(dataset['labels']),1)
        data, labels = self.suffle(data, labels)
        img_length = 80
        return data.reshape(-1,3, img_length, img_length).transpose([0,2,3,1]),labels

    def suffle(self, data, labels):
        s = np.arange(data.shape[0])
        np.random.shuffle(s)
        return data[s], labels[s]

    def get_data(self):
        dataset = self.read_data_set()
        images, labels = self.get_data_from_data_set(dataset)
        return images, labels
