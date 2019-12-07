from DataReader import DataReader
from FeatureExtractor import FeatureExtractor
from DataManager import DataManager
from Classifiers import Classifiers
import pandas as pd
import numpy as np

data_reader = DataReader()
images, labels = data_reader.get_data()

feature_extractor = FeatureExtractor(images, labels)
feature_descriptors = feature_extractor.get_all_feature_descriptors()

classifier_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
 "Naive Bayes", "Logistic Regression", "QDA"]
feature_descriptors_names = {"Histogram Of Gradient", "Haralick", "HuMoments", "Color Histogram"}
list_accuracies = []

#red is the best feature descriptor for the classifier
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: red' if v else '' for v in is_max]

def highlight_max2(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)

for fd in feature_descriptors:
    data_manager = DataManager()
    x_train, x_test, y_train, y_test = data_manager.split(fd, labels, 80)
    classifiers = Classifiers(x_train, x_test, y_train, y_test)
    accuracies = classifiers.run_all()
    list_accuracies.append(accuracies)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
df = pd.DataFrame(list_accuracies, columns=classifier_names)
df.index = feature_descriptors_names
df = df.T
df.style.apply(highlight_max)
df.style.apply(highlight_max2, color='darkorange', axis=None)
print("###############################################################################################################")
print(df)
