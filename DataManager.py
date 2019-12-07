import numpy as np

class DataManager:
    def __init__(self):
        pass

    def split(self, features, labels, percentage):
        data_frame = np.hstack((features, labels))
        partition = int(len(features)*percentage/100)
        x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
        y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()
        return x_train, x_test, y_train, y_test
