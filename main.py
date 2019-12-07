from DataReader import DataReader
from FeatureExtractor import FeatureExtractor
from DataManager import DataManager
from Classifiers import Classifiers

data_reader = DataReader()
images, labels = data_reader.get_data()

feature_extractor = FeatureExtractor(images, labels)
hog_features = feature_extractor.feature_hog()

data_manager = DataManager()
x_train, x_test, y_train, y_test = data_manager.split(hog_features, labels, 80)

classifiers = Classifiers(x_train, x_test, y_train, y_test)
classifiers.run_all()
